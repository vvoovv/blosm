from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidPntsError
from py3dtiles.tileset.content.feature_table import (
    FeatureTable,
    FeatureTableBody,
    FeatureTableHeader,
)
from py3dtiles.typing import FeatureTableHeaderDataType

if TYPE_CHECKING:
    from .tile_content import TileContentHeader


class SemanticPoint(Enum):
    NONE = 0
    POSITION = 1
    POSITION_QUANTIZED = 2
    RGBA = 3
    RGB = 4
    RGB565 = 5
    NORMAL = 6
    NORMAL_OCT16P = 7
    BATCH_ID = 8


class SemanticCategory(Enum):
    NONE = 0
    POSITION = 1
    COLOR = 2
    NORMAL = 3
    BATCH = 4


SEMANTIC_CATEGORY_MAP = {
    SemanticPoint.NONE: SemanticCategory.NONE,
    SemanticPoint.POSITION: SemanticCategory.POSITION,
    SemanticPoint.POSITION_QUANTIZED: SemanticCategory.POSITION,
    SemanticPoint.RGB: SemanticCategory.COLOR,
    SemanticPoint.RGBA: SemanticCategory.COLOR,
    SemanticPoint.RGB565: SemanticCategory.COLOR,
    SemanticPoint.NORMAL: SemanticCategory.NORMAL,
    SemanticPoint.NORMAL_OCT16P: SemanticCategory.NORMAL,
    SemanticPoint.BATCH_ID: SemanticCategory.BATCH,
}

SEMANTIC_TYPE_MAP = {
    SemanticPoint.POSITION: np.float32,
    SemanticPoint.POSITION_QUANTIZED: np.uint16,
    SemanticPoint.RGB: np.uint8,
    SemanticPoint.RGBA: np.uint8,
    SemanticPoint.RGB565: np.uint16,
    SemanticPoint.NORMAL: np.float32,
    SemanticPoint.NORMAL_OCT16P: np.uint8,
}

SEMANTIC_DIMENSION_MAP = {
    SemanticPoint.POSITION: 3,
    SemanticPoint.POSITION_QUANTIZED: 3,
    SemanticPoint.RGB: 3,
    SemanticPoint.RGBA: 4,
    SemanticPoint.RGB565: 1,
    SemanticPoint.NORMAL: 3,
    SemanticPoint.NORMAL_OCT16P: 2,
}

SEMANTIC_ITEM_SIZE_MAP = {
    semantic: SEMANTIC_DIMENSION_MAP[semantic]
    * np.dtype(SEMANTIC_TYPE_MAP[semantic]).itemsize
    for semantic in SEMANTIC_TYPE_MAP
}


def check_semantic_type(semantic: SemanticPoint, category: SemanticCategory) -> None:
    """
    This function checks if the category of the semantic is the same as the parameter category.
    If not, it raises an InvalidPntsError exception.
    """
    if SEMANTIC_CATEGORY_MAP[semantic] != category:
        raise InvalidPntsError(
            f"The category of {semantic} is {SEMANTIC_CATEGORY_MAP[semantic]}, it cannot be used to set {category}"
        )


def check_array_size(
    array: npt.NDArray[np.float32 | np.uint16 | np.uint8],
    semantic: SemanticPoint,
    nb_points: int,
) -> None:
    """
    This function checks if the size of the given array is correct according semantic and nb_points.
    If not, it raises an InvalidPntsError exception.
    """
    if len(array) != nb_points * SEMANTIC_DIMENSION_MAP[semantic]:
        raise InvalidPntsError(
            f"The array {SEMANTIC_CATEGORY_MAP[semantic]} has a wrong size. Expecting a size of "
            f"{nb_points * SEMANTIC_DIMENSION_MAP[semantic]}, got {len(array)}"
        )


class PntsFeatureTableHeader(FeatureTableHeader):
    def __init__(self) -> None:
        # point semantics
        self.positions = SemanticPoint.POSITION
        self.positions_offset = 0
        # Needed when the position is quantized
        self.quantized_volume_offset: npt.NDArray[np.float32] | None = None
        self.quantized_volume_scale: npt.NDArray[np.float32] | None = None

        self.colors = SemanticPoint.NONE
        self.colors_offset = 0
        self.constant_rgba: npt.NDArray[np.uint8] | None = None

        self.normal = SemanticPoint.NONE
        self.normal_offset = 0

        # global semantics
        self.points_length = 0
        self.rtc: tuple[float, float, float] | None = None

    def to_array(self) -> npt.NDArray[np.uint8]:
        jsond = self.to_json()
        json_str = json.dumps(jsond).replace(" ", "")
        n = len(json_str) + 28
        if n % 8 != 0:
            json_str += " " * (8 - n % 8)
        return np.frombuffer(json_str.encode("utf-8"), dtype=np.uint8)

    def to_json(self) -> FeatureTableHeaderDataType:
        # length
        jsond: FeatureTableHeaderDataType = {"POINTS_LENGTH": self.points_length}

        # RTC (Relative To Center)
        if self.rtc is not None:
            jsond["RTC_CENTER"] = self.rtc

        # positions
        offset = {"byteOffset": self.positions_offset}
        if self.positions == SemanticPoint.POSITION:
            jsond["POSITION"] = offset
        elif self.positions == SemanticPoint.POSITION_QUANTIZED:
            jsond["POSITION_QUANTIZED"] = offset
            if self.quantized_volume_offset is None:
                raise InvalidPntsError(
                    "If the position semantic is SemanticPoint.POSITION_QUANTIZED, the attribute quantized_volume_offset cannot be None."
                )
            if self.quantized_volume_scale is None:
                raise InvalidPntsError(
                    "If the position semantic is SemanticPoint.POSITION_QUANTIZED, the attribute quantized_volume_scale cannot be None."
                )
            jsond["QUANTIZED_VOLUME_OFFSET"] = list(self.quantized_volume_offset)
            jsond["QUANTIZED_VOLUME_SCALE"] = list(self.quantized_volume_scale)

        # colors
        offset = {"byteOffset": self.colors_offset}
        if self.colors == SemanticPoint.RGB:
            jsond["RGB"] = offset
        elif self.colors == SemanticPoint.RGBA:
            jsond["RGBA"] = offset
        elif self.colors == SemanticPoint.RGB565:
            jsond["RGB565"] = offset

        if self.constant_rgba is not None:
            # cannot give the shape of self.constant_rgba to mypy
            constant_rgba_tuple: tuple[int, int, int, int] = tuple(self.constant_rgba)
            jsond["CONSTANT_RGBA"] = constant_rgba_tuple

        # normal
        offset = {"byteOffset": self.normal_offset}
        if self.positions == SemanticPoint.NORMAL:
            jsond["NORMAL"] = offset
        elif self.positions == SemanticPoint.NORMAL_OCT16P:
            jsond["NORMAL_OCT16P"] = offset

        return jsond

    @staticmethod
    def from_semantic(
        position_semantic: Literal[
            SemanticPoint.POSITION, SemanticPoint.POSITION_QUANTIZED
        ],
        color_semantic: Literal[
            SemanticPoint.RGB, SemanticPoint.RGBA, SemanticPoint.RGB565
        ]
        | None,
        normal_semantic: Literal[SemanticPoint.NORMAL, SemanticPoint.NORMAL_OCT16P]
        | None,
        nb_points: int,
        quantized_volume_offset: npt.NDArray[np.float32] | None = None,
        quantized_volume_scale: npt.NDArray[np.float32] | None = None,
        constant_rgba: npt.NDArray[np.uint8] | None = None,
    ) -> PntsFeatureTableHeader:
        fth = PntsFeatureTableHeader()
        fth.points_length = nb_points

        fth.positions = position_semantic
        fth.positions_offset = 0
        next_offset = SEMANTIC_ITEM_SIZE_MAP[fth.positions] * nb_points
        if fth.positions == SemanticPoint.POSITION_QUANTIZED:
            if quantized_volume_offset is None:
                raise InvalidPntsError(
                    "If the position is quantized, quantized_volume_offset should be set."
                )
            if quantized_volume_scale is None:
                raise InvalidPntsError(
                    "If the position is quantized, quantized_volume_scale should be set."
                )

            fth.quantized_volume_offset = quantized_volume_offset
            fth.quantized_volume_scale = quantized_volume_scale

        if color_semantic:
            fth.colors = color_semantic
            fth.colors_offset = next_offset
            next_offset += SEMANTIC_ITEM_SIZE_MAP[fth.colors] * nb_points

        if constant_rgba:
            fth.constant_rgba = constant_rgba

        if normal_semantic:
            fth.normal = normal_semantic
            fth.colors_offset = next_offset

        return fth

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> PntsFeatureTableHeader:
        jsond = json.loads(array.tobytes().decode("utf-8"))
        fth = PntsFeatureTableHeader()

        # points length
        if "POINTS_LENGTH" in jsond:
            fth.points_length = jsond["POINTS_LENGTH"]
        else:
            raise InvalidPntsError(
                "The pnts feature table json in the array must have a POINTS_LENGTH entry."
            )

        # search position
        if "POSITION" in jsond:
            fth.positions = SemanticPoint.POSITION
            fth.positions_offset = jsond["POSITION"]["byteOffset"]
        elif "POSITION_QUANTIZED" in jsond:
            fth.positions = SemanticPoint.POSITION_QUANTIZED
            fth.positions_offset = jsond["POSITION_QUANTIZED"]["byteOffset"]

            if (
                "QUANTIZED_VOLUME_OFFSET" not in jsond
                or "QUANTIZED_VOLUME_SCALE" not in jsond
            ):
                raise InvalidPntsError(
                    "When the position is quantized, the pnts feature table json in the array"
                    "must have a QUANTIZED_VOLUME_OFFSET and QUANTIZED_VOLUME_SCALE entry."
                )

            fth.quantized_volume_offset = jsond["QUANTIZED_VOLUME_OFFSET"]
            fth.quantized_volume_offset = jsond["QUANTIZED_VOLUME_SCALE"]
        else:
            raise InvalidPntsError(
                "The pnts feature table json in the array must have a position entry "
                "(either POSITION or POSITION_QUANTIZED"
            )

        # search colors
        if "RGB" in jsond:
            fth.colors = SemanticPoint.RGB
            fth.colors_offset = jsond["RGB"]["byteOffset"]
        elif "RGBA" in jsond:
            fth.colors = SemanticPoint.RGBA
            fth.colors_offset = jsond["RGBA"]["byteOffset"]
        elif "RGB565" in jsond:
            fth.colors = SemanticPoint.RGB565
            fth.colors_offset = jsond["RGB565"]["byteOffset"]
        else:
            fth.colors = SemanticPoint.NONE
            fth.colors_offset = 0

        # search normal
        if "NORMAL" in jsond:
            fth.normal = SemanticPoint.NORMAL
            fth.normal_offset = jsond["NORMAL"]["byteOffset"]
        elif "NORMAL_OCT16P" in jsond:
            fth.normal = SemanticPoint.NORMAL_OCT16P
            fth.normal_offset = jsond["NORMAL_OCT16P"]["byteOffset"]

        # RTC (Relative To Center)
        fth.rtc = jsond.get("RTC_CENTER", None)

        return fth


class PntsFeatureTableBody(FeatureTableBody):
    def __init__(self) -> None:
        self.position: npt.NDArray[np.float32 | np.uint8] = np.array(
            [], dtype=np.float32
        )

        self.color: npt.NDArray[np.uint8 | np.uint16] | None = None
        self.normal: npt.NDArray[np.float32 | np.uint8] | None = None

    @classmethod
    def from_array(
        cls, feature_table_header: PntsFeatureTableHeader, array: npt.NDArray[np.uint8]
    ) -> PntsFeatureTableBody:
        feature_table_body = cls()

        nb_points = feature_table_header.points_length

        # extract positions
        feature_table_body.position = cls._fetch_semantic_from_array(  # type: ignore [assignment] # there is an error on dtype, but _fetch_semantic_from_array runs check_semantic_type.
            array,
            feature_table_header.positions,
            feature_table_header.positions_offset,
            nb_points,
            SemanticCategory.POSITION,
        )

        # extract colors
        if feature_table_header.colors != SemanticPoint.NONE:
            feature_table_body.color = cls._fetch_semantic_from_array(  # type: ignore [assignment] # there is an error on dtype, but _fetch_semantic_from_array runs check_semantic_type.
                array,
                feature_table_header.colors,
                feature_table_header.colors_offset,
                nb_points,
                SemanticCategory.COLOR,
            )

        # extract normals
        if feature_table_header.normal != SemanticPoint.NONE:
            feature_table_body.normal = cls._fetch_semantic_from_array(  # type: ignore [assignment] # there is an error on dtype, but _fetch_semantic_from_array runs check_semantic_type.
                array,
                feature_table_header.normal,
                feature_table_header.colors_offset,
                nb_points,
                SemanticCategory.NORMAL,
            )

        return feature_table_body

    def to_array(self) -> npt.NDArray[np.uint8]:
        position_array = self.position.view(np.uint8)
        length_array = len(position_array)

        if self.color is not None:
            color_array = self.color.view(np.uint8)
            length_array += len(color_array)
        else:
            color_array = np.array([], dtype=np.uint8)

        if self.normal is not None:
            normal_array = self.normal.view(np.uint8)
            length_array += len(normal_array)
        else:
            normal_array = np.array([], dtype=np.uint8)

        padding_str = " " * ((8 - length_array) % 8 % 8)
        padding = np.frombuffer(padding_str.encode("utf-8"), dtype=np.uint8)

        return np.concatenate((position_array, color_array, normal_array, padding))

    @staticmethod
    def _fetch_semantic_from_array(
        array: npt.NDArray[np.uint8],
        semantic: SemanticPoint,
        offset: int,
        nb_points: int,
        category: SemanticCategory,
    ) -> npt.NDArray[np.float32 | np.uint16 | np.uint8]:
        check_semantic_type(semantic, category)

        semantic_array = array[
            offset : offset + SEMANTIC_ITEM_SIZE_MAP[semantic] * nb_points
        ].view(SEMANTIC_TYPE_MAP[semantic])

        check_array_size(semantic_array, semantic, nb_points)

        return semantic_array


class PntsFeatureTable(FeatureTable[PntsFeatureTableHeader, PntsFeatureTableBody]):
    def __init__(self) -> None:
        self.header = PntsFeatureTableHeader()
        self.body = PntsFeatureTableBody()

    def nb_points(self) -> int:
        return self.header.points_length

    def to_array(self) -> npt.NDArray[np.uint8]:
        fth_arr = self.header.to_array()
        ftb_arr = self.body.to_array()
        return np.concatenate((fth_arr, ftb_arr))

    @staticmethod
    def from_array(
        tile_header: TileContentHeader, array: npt.NDArray[np.uint8]
    ) -> PntsFeatureTable:
        # build feature table header
        feature_table_header = PntsFeatureTableHeader.from_array(
            array[: tile_header.ft_json_byte_length]
        )

        feature_table_body = PntsFeatureTableBody.from_array(
            feature_table_header,
            array[
                tile_header.ft_json_byte_length : tile_header.ft_json_byte_length
                + tile_header.ft_bin_byte_length
            ],
        )

        # build feature table
        feature_table = PntsFeatureTable()
        feature_table.header = feature_table_header
        feature_table.body = feature_table_body

        return feature_table

    @staticmethod
    def from_features(
        feature_table_header: PntsFeatureTableHeader,
        position_array: npt.NDArray[np.float32 | np.uint8],
        color_array: npt.NDArray[np.uint8 | np.uint16] | None = None,
        normal_position: npt.NDArray[np.float32 | np.uint8] | None = None,
    ) -> PntsFeatureTable:
        feature_table = PntsFeatureTable()
        feature_table.header = feature_table_header
        nb_points = feature_table.header.points_length

        # set the position array
        check_semantic_type(feature_table.header.positions, SemanticCategory.POSITION)
        check_array_size(position_array, feature_table.header.positions, nb_points)
        feature_table.body.position = position_array

        # set the color array
        if feature_table.header.colors != SemanticPoint.NONE:
            if color_array is None:
                raise InvalidPntsError(
                    f"The argument color_array cannot be None "
                    f"if the color has a semantic of {feature_table.header.colors} in the feature_table_header"
                )
            check_semantic_type(feature_table.header.colors, SemanticCategory.COLOR)
            check_array_size(color_array, feature_table.header.colors, nb_points)
        feature_table.body.color = color_array

        # set the normal array
        if feature_table.header.normal != SemanticPoint.NONE:
            if normal_position is None:
                raise InvalidPntsError(
                    f"The argument normal_array cannot be None "
                    f"if the color has a semantic of {feature_table.header.normal} in the feature_table_header"
                )
            check_semantic_type(feature_table.header.positions, SemanticCategory.NORMAL)
            check_array_size(normal_position, feature_table.header.normal, nb_points)
        feature_table.body.normal = normal_position

        return feature_table

    def get_feature_at(
        self, index: int
    ) -> tuple[
        npt.NDArray[np.float32 | np.uint8],
        npt.NDArray[np.uint8 | np.uint16] | None,
        npt.NDArray[np.float32 | np.uint8] | None,
    ]:
        position = self.get_feature_position_at(index)
        color = self.get_feature_color_at(index)
        normal = self.get_feature_normal_at(index)

        return position, color, normal

    def get_feature_position_at(self, index: int) -> npt.NDArray[np.float32 | np.uint8]:
        if index >= self.nb_points():
            raise IndexError(
                f"The index {index} is out of range. The number of point is {self.nb_points()}"
            )

        check_semantic_type(self.header.positions, SemanticCategory.POSITION)

        dimension = SEMANTIC_DIMENSION_MAP[self.header.positions]
        return self.body.position[dimension * index : dimension * (index + 1)]

    def get_feature_color_at(
        self, index: int
    ) -> npt.NDArray[np.uint8 | np.uint16] | None:
        if index >= self.nb_points():
            raise IndexError(
                f"The index {index} is out of range. The number of point is {self.nb_points()}"
            )

        if self.header.colors == SemanticPoint.NONE:
            return self.header.constant_rgba

        if self.body.color is None:
            raise InvalidPntsError(
                "The feature table body color shouldn't be None "
                f"if self.header.colors is {self.header.colors}."
            )

        check_semantic_type(self.header.colors, SemanticCategory.COLOR)

        dimension = SEMANTIC_DIMENSION_MAP[self.header.colors]
        return self.body.color[dimension * index : dimension * (index + 1)]

    def get_feature_normal_at(
        self, index: int
    ) -> npt.NDArray[np.float32 | np.uint8] | None:
        if index >= self.nb_points():
            raise IndexError(
                f"The index {index} is out of range. The number of point is {self.nb_points()}"
            )

        if self.header.normal == SemanticPoint.NONE:
            return None

        if self.body.normal is None:
            raise InvalidPntsError(
                "The feature table body normal shouldn't be None "
                f"if self.header.colors is {self.header.normal}."
            )

        check_semantic_type(self.header.normal, SemanticCategory.NORMAL)

        dimension = SEMANTIC_DIMENSION_MAP[self.header.normal]
        return self.body.normal[dimension * index : dimension * (index + 1)]
