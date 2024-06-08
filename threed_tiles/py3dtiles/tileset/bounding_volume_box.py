from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import TilerException
from py3dtiles.typing import BoundingVolumeBoxDictType

from .bounding_volume import BoundingVolume

if TYPE_CHECKING:
    from typing_extensions import Self

    from .tile import Tile

# In order to prevent the appearance of ghost newline characters ("\n")
# when printing a numpy.array (mainly self._box in this file):
np.set_printoptions(linewidth=500)


class BoundingVolumeBox(BoundingVolume[BoundingVolumeBoxDictType]):
    """
    A box bounding volume as defined in the 3DTiles specifications i.e. an
    array of 12 numbers that define an oriented bounding box:
    - The first three elements define the x, y, and z values for the
    center of the box.
    - The next three elements (with indices 3, 4, and 5) define the x axis
    direction and half-length.
    - The next three elements (with indices 6, 7, and 8) define the y axis
    direction and half-length.
    - The last three elements (indices 9, 10, and 11) define the z axis
    direction and half-length.

    Note that, by default, a box bounding volume doesn't need to be aligned
    with the coordinate axis. Still in general, computing the box bounding
    volume of two box bounding volumes won't necessarily yield a box that is
    aligned with the coordinate axis (although this computation might require
    some fitting algorithm e.g. the principal component analysis method.
    Yet in sake of simplification (and numerical efficiency), when asked to
    "add" (i.e. to find the enclosing box of) two (or more) box bounding
    volumes this class resolves to compute the "canonical" fitting/enclosing
    box i.e. a box that is parallel to the coordinate axis.
    """

    def __init__(self) -> None:
        super().__init__()
        self._box: npt.NDArray[np.float64] | None = None

    @classmethod
    def from_dict(cls, bounding_volume_box_dict: BoundingVolumeBoxDictType) -> Self:
        bounding_volume_box = cls()
        bounding_volume_box.set_from_list(bounding_volume_box_dict["box"])

        bounding_volume_box.set_properties_from_dict(bounding_volume_box_dict)

        return bounding_volume_box

    def get_center(self) -> npt.NDArray[np.float64]:
        if self._box is None:
            raise AttributeError("Bounding Volume Box is not defined.")

        return self._box[0:3]

    def translate(self, offset: npt.NDArray[np.float64]) -> None:
        """
        Translate the box center with the given offset "vector"
        :param offset: the 3D vector by which the box should be translated
        """
        if self._box is None:
            raise AttributeError("Bounding Volume Box is not defined.")

        self._box[:3] += offset[:3]

    def transform(self, transform: npt.NDArray[np.float64]) -> None:
        """
        Apply the provided transformation matrix (4x4) to the box
        :param transform: transformation matrix (4x4) to be applied
        """
        if self._box is None:
            raise AttributeError("Bounding Volume Box is not defined.")

        transform = transform.flatten()

        # FIXME: the following code only uses the first three coordinates
        # of the transformation matrix (and basically ignores the fourth
        # column of transform). This looks like some kind of mistake...
        rotation = np.array([transform[0:3], transform[4:7], transform[8:11]])

        center = self._box[0:3]
        x_half_axis = self._box[3:6]
        y_half_axis = self._box[6:9]
        z_half_axis = self._box[9:12]

        # Apply the rotation part to each element
        new_center = rotation.dot(center)
        new_x_half_axis = rotation.dot(x_half_axis)
        new_y_half_axis = rotation.dot(y_half_axis)
        new_z_half_axis = rotation.dot(z_half_axis)
        self._box = np.concatenate(
            (new_center, new_x_half_axis, new_y_half_axis, new_z_half_axis)
        )
        offset = transform[12:15]
        self.translate(offset)

    def is_box(self) -> bool:
        return True

    def set_from_list(self, box_list: npt.ArrayLike) -> None:
        box = np.array(box_list, dtype=float)

        valid, reason = BoundingVolumeBox.is_valid(box)
        if not valid:
            raise ValueError(reason)
        self._box = box

    def set_from_points(self, points: list[npt.NDArray[np.float64]]) -> None:
        box = BoundingVolumeBox.get_box_array_from_point(points)

        valid, reason = BoundingVolumeBox.is_valid(box)
        if not valid:
            raise ValueError(reason)
        self._box = box

    def set_from_mins_maxs(self, mins_maxs: npt.NDArray[np.float64]) -> None:
        """
        :param mins_maxs: the array [x_min, y_min, z_min, x_max, y_max, z_max]
                          that is the boundaries of the box along each
                          coordinate axis
        """
        self._box = BoundingVolumeBox.get_box_array_from_mins_maxs(mins_maxs)

    def get_corners(self) -> list[npt.NDArray[np.float64]]:
        """
        :return: the corners (3D points) of the box as a list
        """
        if self._box is None:
            raise AttributeError("Bounding Volume Box is not defined.")

        center, x_half_axis, y_half_axis, z_half_axis = self._box.reshape([-1, 3])

        x_axis = x_half_axis * 2
        y_axis = y_half_axis * 2
        z_axis = z_half_axis * 2

        # The eight cornering points of the box
        origin = center - x_half_axis - y_half_axis - z_half_axis

        ox = origin + x_axis
        oy = origin + y_axis
        oz = origin + z_axis
        oxy = ox + y_axis
        oxz = ox + z_axis
        oyz = oy + z_axis
        oxyz = oxy + z_axis

        return [origin, ox, oy, oxy, oz, oxz, oyz, oxyz]

    def get_canonical_as_array(self) -> npt.NDArray[np.float64]:
        """
        :return: the smallest enclosing box (as an array) that is parallel
                 to the coordinate axis
        """
        return BoundingVolumeBox.get_box_array_from_point(self.get_corners())

    def add(self, other: BoundingVolume[Any]) -> None:
        """
        Compute the 'canonical' bounding volume fitting this bounding volume
        together with the added bounding volume. Again (refer above to the
        class definition) the computed fitting bounding volume is generically
        not the smallest one (due to its alignment with the coordinate axis).
        :param other: another box bounding volume to be added with this one
        """
        if not isinstance(other, BoundingVolumeBox):
            raise NotImplementedError(
                "The add method works only with BoundingVolumeBox"
            )

        if self._box is None:
            # Then it is safe to overwrite
            self._box = copy.deepcopy(other._box)
            return

        corners = self.get_corners() + other.get_corners()
        self.set_from_points(corners)

    def sync_with_children(self, owner: Tile) -> None:
        # We reset to some dummy state of this Bounding Volume Box so we
        # can add up in place the boxes of the owner's children
        # If there is no child, no modifications are done.
        for child in owner.children:
            if child.bounding_volume is None:
                raise TilerException("Child should have a bounding volume.")

            bounding_volume = copy.deepcopy(child.bounding_volume)
            bounding_volume.transform(child.transform)
            if not bounding_volume.is_box():
                raise TilerException(
                    "All children must also have a box as bounding volume "
                    "if the parent has a bounding box"
                )
            self.add(bounding_volume)

    def to_dict(self) -> BoundingVolumeBoxDictType:
        if self._box is None:
            raise AttributeError("Bounding Volume Box is not defined.")

        dict_data: BoundingVolumeBoxDictType = {"box": list(self._box)}
        return self.add_root_properties_to_dict(dict_data)

    @staticmethod
    def get_box_array_from_mins_maxs(
        mins_maxs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        :param mins_maxs: the list [x_min, y_min, z_min, x_max, y_max, z_max]
                          that is the boundaries of the box along each
                          coordinate axis
        :return: the smallest box (as an array, as opposed to a
                BoundingVolumeBox instance) that encloses the given list of
                (3D) points and that is parallel to the coordinate axis.
        """
        x_min = mins_maxs[0]
        x_max = mins_maxs[3]
        y_min = mins_maxs[1]
        y_max = mins_maxs[4]
        z_min = mins_maxs[2]
        z_max = mins_maxs[5]
        new_center = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        )
        new_x_half_axis = np.array([(x_max - x_min) / 2, 0, 0])
        new_y_half_axis = np.array([0, (y_max - y_min) / 2, 0])
        new_z_half_axis = np.array([0, 0, (z_max - z_min) / 2])

        return np.concatenate(
            (new_center, new_x_half_axis, new_y_half_axis, new_z_half_axis)
        )

    @staticmethod
    def get_box_array_from_point(
        points: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        :param points: a list of 3D points
        :return: the smallest box (as an array, as opposed to a
                BoundingVolumeBox instance) that encloses the given list of
                (3D) points and that is parallel to the coordinate axis.
        """
        return BoundingVolumeBox.get_box_array_from_mins_maxs(
            np.array(
                [
                    min(c[0] for c in points),
                    min(c[1] for c in points),
                    min(c[2] for c in points),
                    max(c[0] for c in points),
                    max(c[1] for c in points),
                    max(c[2] for c in points),
                ]
            )
        )

    @staticmethod
    def is_valid(box: npt.NDArray[np.float64]) -> tuple[bool, str]:
        if box is None:
            return False, "Bounding Volume Box is not defined."
        if box.ndim != 1:
            return False, "Bounding Volume Box has wrong dimensions."
        if box.shape[0] != 12:
            return False, "Warning: Bounding Volume Box must have 12 elements."
        return True, ""
