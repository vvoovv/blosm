# From https://github.com/DataDog/piecewise
# 
# SD 3-Clause License
# Copyright (c) 2017, Datadog, Inc.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# std
from collections import namedtuple
import heapq

# 3p
import numpy as np


## Function to learn and plot piecewise regressions.


def piecewise(t, v, min_stop_frac=0.03):
    """ Fits a piecewise (aka "segmented") regression.
    Params:
        t (listlike of ints or floats): independent/predictor variable values
        v (listlike of ints or floats): dependent/outcome variable values
        min_stop_frac (float between 0 and 1): the fraction of total error that
            a merge must account for to be considered big enough to stop merging;
            the default is usually adequate, but this may be increased to make
            merging more aggressive (leading to fewer segments in the result)
    Returns:
        A FittedModel object that can be used for interpolation and extrapolation.
    """
    # Validate the inputs, and force t and v to be np.arrays sorted in
    # ascending t order.
    t, v = _preprocess(t, v)

    # Initialize the segments.
    init_segments, merges = _get_initial_segments_and_merges(t, v)
    seg_tracker = SegmentTracker(init_segments)

    # Use a min heap to track potential merges. At the top of the heap
    # will be the next best merge (smallest increase in error).
    heapq.heapify(merges)

    # Greedily make the next best merge until we've merged everything together into
    # one segment.
    cum_cost, biggest_cost_increase = 0.0, 0.0

    # list of merges we need to undo to return to last best configuration
    merges_since_best = []

    while len(seg_tracker) > 1:
        # Identify the next merge to be executed.
        next_merge = _get_next_merge(merges, seg_tracker)

        # If the next merge increases the error by a larger amount than any
        # merge so far, remember the current state (which might end up being the
        # "best"). To prevent stopping too early (for example, in cases where
        # there should only be one segment), use min_stop_frac to keep on
        # remembering the current state as the best state if no single
        # merge has accounted for a significant part of the total error.
        cum_cost += next_merge.cost
        cost_increase = next_merge.cost - biggest_cost_increase
        biggest_cost_increase = max(biggest_cost_increase, cost_increase)
        if biggest_cost_increase < min_stop_frac*cum_cost or \
                cost_increase == biggest_cost_increase:
            merges_since_best = [next_merge]
        else:
            merges_since_best.append(next_merge)

        # Execute the next merge.
        # Update segments, replacing the two old ones with the one new one.
        seg_tracker.apply_merge(next_merge)
        
        # Add new potential merges.
        neighbors = seg_tracker.get_neighbors(next_merge.new_seg)
        for neighbor in neighbors:
            left_seg, right_seg = sorted([next_merge.new_seg, neighbor])
            heapq.heappush(merges, _make_merge(t, v, left_seg, right_seg))

    if biggest_cost_increase < min_stop_frac*cum_cost:
        # This path is needed for the case where there is only one segment, because
        # merges_since_best isn't updated after merging in the loop above.
        merges_since_best = []
    
    for merge in reversed(merges_since_best):
        seg_tracker.unapply_merge(merge)

    fitted_segments = [
        FittedSegment(t[seg.start_index], t[min(seg.end_index, len(t)-1)], seg.coeffs)
        for seg in seg_tracker.segments
    ]
    return FittedModel(fitted_segments)


## Data structures used for representing the fitted model returned by `piecewise()`.


class FittedSegment(namedtuple('FittedSegment',
    [
        'start_t',  # (float) first t value to which this segment applies
        'end_t',    # (float) first t value to which this segment no longer applies
        'coeffs'    # (tuple of floats) regression coefficients
    ]
)):
    def predict(self, t_new, out=None, where=True):
        return _predict(self.coeffs, t_new, out=out, where=where)


class FittedModel(object):
    """ Completely defines the result of a piecewise regression.
    The `segments` attribute contains a list of FittedSegments.
    """

    def __init__(self, fitted_segments):
        self.segments = fitted_segments
        self._starts = [fs.start_t for fs in fitted_segments]

    def __repr__(self):
        return 'FittedModel with segments:\n' + '\n'.join(
            ['* ' + seg.__repr__() for seg in self.segments]
        )

    def predict(self, t_new):
        """ Use the segments in this model to predict the v value for new t values.
        Params:
            t_new (scalar or array like): t values for which predictions should be made
        Returns:
            scalar or array like of predictions
        """
        if len(self.segments) == 1:
            return self.segments[0].predict(t_new)

        t_array = np.asanyarray(t_new)
        seg_index = np.digitize(t_array, [s.end_t for s in self.segments[:-1]])
        if seg_index.shape == (): # t_new is a scalar or 0-dimensional array
            return self.segments[seg_index].predict(t_new)
        else:
            v_hats = np.empty_like(t_array, dtype=np.double)
            for i, segment in enumerate(self.segments):
                segment.predict(t_array, out=v_hats, where=seg_index == i)
            return v_hats


## Data structures used during the fitting of the regression in `piecewise()`.


# Segment represents a time range and a linear regression fit through it.
Segment = namedtuple('Segment',
    [
        'start_index',  # (int) zero-based index of start time
        'end_index',    # (int) zero-based index of non-inclusive end time
        'coeffs',       # (tuple of floats) regression coefficients
        'error',        # (float) the total error in the segment
        'cov_data',     # (np.array of floats) incremental covariance data
    ]
)


# Merge represents a potential merge of two neighboring segments.
Merge = namedtuple('Merge',
    [
        'cost',       # (float) increase in sum of squared error that would result from executing this merge
        'left_seg',   # (Segment)
        'right_seg',  # (Segment)
        'new_seg'     # (Segment) the Segment that would result from merging combining left_seg and right_seg
    ]
)


class SegmentTracker(object):
    """ Utility class for tracking the state of the piecewise regression (i.e.,
    what are the current segments based on the set of merges that have been
    executed so far).
    """

    def __init__(self, segments):
        # Assume segments are sorted
        starts = np.fromiter((s.start_index for s in segments), np.intp, count=len(segments))

        # One position for each original index.
        # About 50% of this space is wasted, but this enables O(1) lookup and
        # replacement by start_index
        self._segments = np.empty(segments[-1].end_index, dtype=object)
        self._segments[starts] = segments

        # Valid mask.  As segments are merged, this mask is updated
        self._valid = np.not_equal(self._segments, None)

        # Previous neighbor lookup
        self._prev = np.zeros_like(self._segments, dtype=np.intp)
        self._prev[starts[1:]] = starts[:-1]

        # Cached length.  Without this, we would need to count _valid every len()
        self._len = len(segments)

    def __len__(self):
        return self._len

    def contains(self, segment):
        """ Returns True if segment is currently valid; False otherwise. """
        # segment at start_index has not been merged away and is still the same
        return self._valid[segment.start_index] \
            and self._segments[segment.start_index] is segment
    
    def get_prev(self, segment):
        """ Returns the left neighbor of segment; None if it is the first. """
        if segment.start_index > 0:
            return self._segments[self._prev[segment.start_index]]
        else:
            return None
    
    def get_next(self, segment):
        """ Returns the right neighbor of segment; None if it is the last. """
        if segment.end_index < len(self._segments):
            return self._segments[segment.end_index]
        else:
            return None

    def get_neighbors(self, segment):
        """ Returns a list of Segments, containing the 0, 1, or 2 segments
        adjacent to the given Segment.
        """
        return (
            s for s in (self.get_prev(segment), self.get_next(segment))
            if s is not None
        )

    def apply_merge(self, merge):
        """ Insert a new segment and remove the two existing segments
        from which it was created.
        """
        right_seg, new_seg = merge.right_seg, merge.new_seg
        self._valid[right_seg.start_index] = False
        _next = self.get_next(right_seg)
        if _next:
            self._prev[_next.start_index] = new_seg.start_index
        self._segments[new_seg.start_index] = new_seg
        self._len -= 1
    
    def unapply_merge(self, merge):
        """ Remove a segment and reinsert the two segments
        from which it was created.
        """
        right_seg, left_seg = merge.right_seg, merge.left_seg
        self._valid[right_seg.start_index] = True
        _next = self.get_next(right_seg)
        if _next:
            self._prev[_next.start_index] = right_seg.start_index
        self._segments[left_seg.start_index] = left_seg
        self._len += 1

    @property
    def segments(self):
        return self._segments[self._valid]


## Helper functions for doing piecewise regression.


def _preprocess(t, v):
    """ Raises an exception if any of the inputs are not valid.
    Otherwise, returns a list of Points, ordered by t.
    """
    # Validate the inputs.
    if len(t) != len(v):
        raise ValueError('`t` and `v` must have the same length.')
    t_arr, v_arr = np.asanyarray(t, dtype=np.double), np.asanyarray(v, dtype=np.double)
    if not np.all(np.isfinite(t)):
        raise ValueError('All values in `t` must be finite.')
    finite_mask = np.isfinite(v_arr)
    if np.sum(finite_mask) < 2:
        raise ValueError('`v` must have at least 2 finite values.')
    t_arr, v_arr = t_arr[finite_mask], v_arr[finite_mask]

    # Order both arrays by t-values.
    sort_order = np.argsort(t_arr)
    t_arr, v_arr = t_arr[sort_order], v_arr[sort_order]

    return t_arr, v_arr


def _get_initial_segments_and_merges(t, v):
    """ Returns a 2-tuple with the lists of initial segments and initial merges.
    Each Segment is of length 1, 2, or 3. They are created by using even-indexed
    points as seeds and attaching odd-indexed points to the neighboring seed with
    the closer v value.
    This initialization procedure exists to decrease the odds of bad initial
    merges. If initial segments were each a single point, then merging any two
    neighboring points would be equally attractive to our algorithm, because the
    squared error of a line fit through any pair of points is zero. However,
    in the case that the data looks like [1, 1, 1, 1, 10, 10, 10, 10], we would
    prefer to avoid the 1 and neighboring 10 from starting out in the same
    segment. This initialization does this by doing initial merges based on
    absolute difference rather than regression error. Unfortunately, there can
    still be suboptimal initializations, as in this case, where the two 1s will
    be initialized in the same segment: [19, 10, 1, 1, -8, -17]
    """

    # creates segments from an array of start, end indices
    def _build_segments(ranges):
        # number of point in range
        n = np.diff(ranges, axis=1).reshape(-1)
        
        # expand ranges in rows, using masked arrays to deal with uneven lengths.
        # indices like these:
        #   [[0, 1], [1, 4], [4, 6]]
        # yield something like this:
        #   [
        #       [v0, --, --],
        #       [v1, v2, v3],
        #       [v4, v5, --],
        #   ]
        max_n = np.max(n)
        indices = np.ma.array(ranges[:,:1] + np.arange(max_n).reshape(1, -1))
        for i in range(1, max_n):
            indices[n == i, i:] = np.ma.masked
        
        segment_t = np.ma.take(t, indices)
        segment_v = np.ma.take(v, indices)
    
        # sum(t), sum(v), unmasked
        st = np.ma.getdata(np.ma.sum(segment_t, axis=1))
        sv = np.ma.getdata(np.ma.sum(segment_v, axis=1))

        # mean(t), mean(v)
        mu_t = (st / n).reshape(-1, 1)
        mu_v = (sv / n).reshape(-1, 1)

        # distance from means
        dt = segment_t - mu_t
        dv = segment_v - mu_v

        # var(t), var(v) and cov(t, v), before division by n, unmasked
        ct = np.ma.getdata(np.ma.sum(dt ** 2, axis=1))
        cv = np.ma.getdata(np.ma.sum(dv ** 2, axis=1))
        ctv = np.ma.getdata(np.ma.sum(dt * dv, axis=1))
        
        # slope and intercept
        # for single point segments (ct == 0), assume slope = 0, intercept = mean(v)
        nonzero_ct = ct > 0
        slope = np.where(nonzero_ct, ctv, 0.0)
        slope = np.divide(slope, ct, out=slope, where=nonzero_ct).reshape(-1, 1)
        intercept = mu_v - slope * mu_t

        # sum of squared errors
        # if n < 3: error = 0
        # elif ct == 0: error = cv
        # else: error = cv - ctv ** 2 / ct

        nonzero_error = n >= 3
        nonzero_ct &= nonzero_error
        error = np.where(nonzero_ct, ctv, 0.0) # 0, 0, ctv
        np.square(error, out=error, where=nonzero_ct) # 0, 0, ctv ** 2
        np.divide(error, ct, out=error, where=nonzero_ct) # 0, 0, ctv ** 2 / ct
        np.subtract(cv, error, out=error, where=nonzero_error) # 0, cv, cv - ctv ** 2 / ct


        return [
            Segment(
                ranges[i, 0], ranges[i, 1], (intercept[i, 0], slope[i, 0]), error[i],
                cov_data
            )
            for i, cov_data in enumerate(np.c_[n, st, sv, ct, cv, ctv])
        ]
    
    # If there are multiple values at the same t, average them and treat them
    # like a single point during initialization. This ensures that all the
    # points with the same t are assigned to the same linear segment.
    unique_t = np.unique(t, return_index=True)[1]
    even_n = len(unique_t) % 2 == 0
    index_ranges = np.c_[unique_t, np.r_[unique_t[1:], len(t)]]
    
    # unique t is pretty common, optimize for that
    averages = v[index_ranges[:,0]]
    long_ranges = np.diff(index_ranges, axis=1).reshape(-1) > 1
    if long_ranges.any():
        averages[long_ranges] = np.fromiter(
            (v[idx[0]:idx[1]].mean() for idx in index_ranges[long_ranges]),
            np.double, count=long_ranges.sum()
        )

    # Pair every other t with the t on its left or on its right, based on which
    # is closer.
    pair_left = np.less(*np.abs(
        np.ediff1d(averages, to_end=np.inf if even_n else None)
    ).reshape(-1, 2).T)
    np.copyto(
        index_ranges[:-1:2, 1],
        index_ranges[1::2, 1],
        where=pair_left
    )
    np.copyto(
        index_ranges[2::2, 0],
        index_ranges[1:-1:2, 0],
        where=~pair_left[:-1 if even_n else None]
    )

    # initial segment ranges are at even indices
    segment_ranges = index_ranges[::2]
    segments = _build_segments(segment_ranges)

    # merge every consecutive segment
    merge_ranges = np.c_[segment_ranges[:-1,0], segment_ranges[1:,1]]
    merge_segments = _build_segments(merge_ranges)
    
    merges = [
        Merge(
            new_seg.error - segments[i].error - segments[i + 1].error,
            segments[i], segments[i + 1], new_seg
        )
        for i, new_seg in enumerate(merge_segments)
    ]

    return segments, merges


def _get_next_merge(merges, segment_tracker):
    """ Returns the valid Merge that has the lowest cost.
    Params:
        merges: a heapified list of Merges
        segment_tracker: a SegmentTracker with the currently valid segments;
            any Merge referencing a Segment not in the tracker is no longer valid
    """
    while True:
        next_merge = heapq.heappop(merges)
        if (segment_tracker.contains(next_merge.left_seg) and
                segment_tracker.contains(next_merge.right_seg)):
            return next_merge


def _make_segment(t, v, left_seg, right_seg):
    """ Returns a Segment that is the merge of left_seg and right_seg,
    starting at left_seg.start_index and ending at the non-inclusive
    right_seg.end_index.
    """
    start_index = left_seg.start_index
    end_index = right_seg.end_index
    cov_data = _merge_cov_data(left_seg.cov_data, right_seg.cov_data)
    coeffs, error = _fit_line(t, v, start_index, end_index, cov_data)
    return Segment(start_index, end_index, coeffs, error, cov_data)


def _make_merge(t, v, left_seg, right_seg):
    """ Returns a Merge combining the left_seg and right_seg Segments.
    """
    new_seg = _make_segment(t, v, left_seg, right_seg)
    cost = new_seg.error - left_seg.error - right_seg.error
    return Merge(cost, left_seg, right_seg, new_seg)


def _merge_cov_data(d1, d2):
    """ Merge covariance data from two segments into a new one.
    See also:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    """
    d3 = d1 + d2
    n1 = d1[0]
    n2 = d2[0]
    n12 = n1 * n2
    n3 = d3[0]
    deltat = (d1[1] * n2 - d2[1] * n1) / n12
    deltav = (d1[2] * n2 - d2[2] * n1) / n12
    d3[3] += deltat ** 2 * n12 / n3
    d3[4] += deltav ** 2 * n12 / n3
    d3[5] += deltat * deltav * n12 / n3
    return d3

def _fit_line(t, v, start_index, end_index, cov_data):
    """ Fits and OLS regression for the set of t and v values in the given index
    range. Returns (coefficients of line, sum of squared error).
    """

    # based on scipy.stats.linregress
    mu_t, mu_v = cov_data[1:3] / cov_data[0]
    ct, cv, ctv = cov_data[3:]
    if ct != 0:
        slope = ctv / ct
        intercept = mu_v - slope * mu_t
        error = cv - ctv ** 2 / ct
    else:
        slope, intercept, error = 0.0, mu_v, cv

    return ((intercept, slope), error)


def _predict(coeffs, t, out=None, where=True):
    """ Given OLS coefficients, predict the corresponding v values for the given
    t values.
    """
    # if out is None, numpy allocates an empty one
    out = np.multiply(t, coeffs[1], out=out, where=where)
    if np.isscalar(out):
        # t was either a scalar or a 0-dimensional array
        # returning a scalar is consistent with numpy arithmetic operations
        return out + coeffs[0]
    else:
        np.add(out, coeffs[0], out=out, where=where)
        return out
