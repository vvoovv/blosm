# This is an implementation of the algorithm described by Adriano Moreira
# and Maribel Yasmina Santos: CONCAVE HULL: A K-NEAREST NEIGHBOURS APPROACH
# FOR THE COMPUTATION OF THE REGION OCCUPIED BY A SET OF POINTS.
# GRAPP 2007 - International Conference on Computer Graphics Theory and
# Applications; pp 61-68.
# https://repositorium.sdum.uminho.pt/bitstream/1822/6429/1/ConcaveHull_ACM_MYS.pdf

# This code has been ported to Python 3+ from
# https://github.com/merowech/java-concave-hull/tree/master , and
# https://github.com/detlevn/QGIS-ConcaveHull-Plugin/blob/master/concavehull.py
#
# Points are assumed to be frozen Vectors of mathutils (mathutils.Vector)
#
from math import pi, atan2

from lib.CompGeom.centerline import pointInPolygon


class ConcaveHull(object):
    def __init__(self):
        self.hull = []
        self.pointSet = set()

    def kNearestNeighbors(self, point, k):
        # The point set <pointSet> is assumed to be small and the nearest neighbors
        # are only searche once for a given set. Therefore, the usage of a KD-tree
        # would not be efficient.
        distanceList = [(s, (s - point).length) for s in self.pointSet]
        nearestList = sorted(distanceList, key=lambda value: value[1])
        # The number of points can't be larger than the number of points in <pointSet>
        return [s[0] for s in nearestList[: min(k, len(self.pointSet))]]

    def concaveHull(self, points, k):
        # points:   Python list of points (frozen Vectors) to process
        # k:        Number of neighbours

        # remove duplicate points
        self.pointSet = list(set(points))

        # k needs to be equal or greater than 3
        kk = max(k, 3)

        # If this is already a concave hull
        if len(self.pointSet) <= 3:
            return self.pointSet

        # make sure that k neighbors can be found
        kk = min(kk, len(self.pointSet) - 1)

        # Find first point (the one with smallest y value) and remove from point set
        firstPoint = min(self.pointSet, key=lambda value: value[1])

        # Add this points as the first vertex of the hull
        self.hull.append(firstPoint)

        # Make the first vertex of the hull to be the current point
        currentPoint = firstPoint

        # Remove the point from the pointSet, to prevent him being among the nearest points
        self.pointSet.remove(firstPoint)

        previousAngle = pi
        step = 2

        while (currentPoint != firstPoint) or (step == 2) and (len(self.pointSet) > 0):
            # After 3 iterations add the first point to pointSet again, otherwise
            # a hull cannot be closed
            if step == 5:
                self.pointSet.append(firstPoint)

            # Get k nearest neighbors of current point*
            kNearestPoints = self.kNearestNeighbors(currentPoint, kk)

            # Sort points by angle clockwise
            clockwisePoints = sorted(
                kNearestPoints,
                key=lambda p: ConcaveHull.angleDifference(
                    previousAngle, ConcaveHull.angle(currentPoint, p)
                ),
            )

            its = True
            i = -1

            # search for the nearest point to which the connecting line does not
            # intersect any existing segment
            while its and (i < len(clockwisePoints) - 1):
                i += 1
                if clockwisePoints[i] == firstPoint:
                    last_point = 1
                else:
                    last_point = 0
                j = 2
                its = False

                while not its and (j < len(self.hull) - last_point):
                    its = ConcaveHull.intersect(
                        (self.hull[step - 2], clockwisePoints[i]),
                        (self.hull[step - 2 - j], self.hull[step - 1 - j]),
                    )
                    j += 1

            # If there is no candidate, to which the connecting line does not
            # intersect any existing segment, so the search for the next
            # candidate fails. The algorithm starts again with an increased
            # number of neighbors.
            if its:
                self.hull = []
                self.pointSet = set()
                return self.concaveHull(points, kk + 1)

            # the first point which complies with the requirements is added to
            # the hull and gets the current point
            currentPoint = clockwisePoints[i]
            self.hull.append(currentPoint)

            # calculate the angle between the last vertex and his precursor,
            # that is the last segment of the hull
            # in reversed direction
            previousAngle = ConcaveHull.angle(self.hull[step - 1], self.hull[step - 2])

            # remove current_point from point_set
            self.pointSet.remove(currentPoint)

            # increment counter
            step += 1

        allInside = True
        i = len(self.pointSet) - 1

        allInside = all(
            pointInPolygon(self.hull, p) in ["IN", "ON"] for p in self.pointSet
        )

        # since at least one point is out of the computed polygon, try again
        # with a higher number of neighbors
        if not allInside:
            return self.concaveHull(points, kk + 1)

        # a valid hull has been constructed
        return self.hull

    @staticmethod
    def angle(p1, p2):
        dp = p2 - p1
        return atan2(dp[1], dp[0])

    @staticmethod
    def angleDifference(a1, a2):
        # Calculate angle difference in clockwise directions as radians
        if a1 > 0 and a2 >= 0 and a1 > a2:
            return abs(a1 - a2)
        elif a1 >= 0 and a2 > 0 and a1 < a2:
            return 2 * pi + a1 - a2
        elif a1 < 0 and a2 <= 0 and a1 < a2:
            return 2 * pi + a1 + abs(a2)
        elif a1 <= 0 and a2 < 0 and a1 > a2:
            return abs(a1 - a2)
        elif a1 <= 0 and 0 < a2:
            return 2 * pi + a1 - a2
        elif a1 >= 0 and 0 >= a2:
            return a1 + abs(a2)
        else:
            return 0.0

    @staticmethod
    def intersect(line1, line2):
        # Returns True if the two given line segments intersect each other,
        # and False otherwise.
        # line1: tuple (p1, p2) of Vector
        # line2: tuple (p3, p4) of Vector
        # return: boolean
        (p1, p2), (p3, p4) = line1, line2
        d1, d2 = (
            (p2[0] - p1[0], p2[1] - p1[1]),
            (p4[0] - p3[0], p4[1] - p3[1]),
        )  # p2-p1, p4-p3
        cross = d1[0] * d2[1] - d2[0] * d1[1]  # d1.cross(d2)
        if cross == 0.0:
            return False
        d3 = (p1[0] - p3[0], p1[1] - p3[1])  # p1-p3
        t1 = (d2[0] * d3[1] - d2[1] * d3[0]) / cross
        t2 = (d1[0] * d3[1] - d1[1] * d3[0]) / cross
        return 0.0 < t1 < 1.0 and 0.0 < t2 < 1
