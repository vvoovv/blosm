from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidPntsError

from .batch_table import BatchTable
from .pnts_feature_table import (
    PntsFeatureTable,
    PntsFeatureTableBody,
    PntsFeatureTableHeader,
    SemanticPoint,
)
from .tile_content import TileContent, TileContentBody, TileContentHeader


class Pnts(TileContent):
    def __init__(self, header: PntsHeader, body: PntsBody) -> None:
        super().__init__()
        self.header: PntsHeader = header
        self.body: PntsBody = body
        self.sync()

    def sync(self) -> None:
        """
        Synchronizes headers with the Pnts body.
        """
        self.header.ft_json_byte_length = len(self.body.feature_table.header.to_array())
        self.header.ft_bin_byte_length = len(self.body.feature_table.body.to_array())
        self.header.bt_json_byte_length = len(self.body.batch_table.header.to_array())
        self.header.bt_bin_byte_length = len(self.body.batch_table.body.to_array())

        self.header.tile_byte_length = (
            PntsHeader.BYTE_LENGTH
            + self.header.ft_json_byte_length
            + self.header.ft_bin_byte_length
            + self.header.bt_json_byte_length
            + self.header.bt_bin_byte_length
        )

    def print_info(self) -> None:
        if self.header:
            th = self.header
            print("Tile Header")
            print("-----------")
            print("Magic Value: ", th.magic_value)
            print("Version: ", th.version)
            print("Tile byte length: ", th.tile_byte_length)
            print("Feature table json byte length: ", th.ft_json_byte_length)
            print("Feature table bin byte length: ", th.ft_bin_byte_length)
        else:
            print("Tile with no header")

        if self.body:
            fth = self.body.feature_table.header
            print("")
            print("Feature Table Header")
            print("--------------------")
            print(fth.to_json())

            # first point data
            if fth.points_length > 0:
                print("")
                print("First point")
                print("-----------")
                (
                    feature_position,
                    feature_color,
                    feature_normal,
                ) = self.body.feature_table.get_feature_at(0)
                print(f"Position: {feature_position}")
                print(f"Color: {feature_color}")
                print(f"Normal: {feature_normal}")
        else:
            print("Tile with no body")

    @staticmethod
    def from_features(
        feature_table_header: PntsFeatureTableHeader,
        position_array: npt.NDArray[np.float32 | np.uint8],
        color_array: npt.NDArray[np.uint8 | np.uint16] | None = None,
        normal_position: npt.NDArray[np.float32 | np.uint8] | None = None,
    ) -> Pnts:
        """
        Creates a Pnts from features defined by pd_type and cd_type.
        """
        pnts_body = PntsBody()
        pnts_body.feature_table = PntsFeatureTable.from_features(
            feature_table_header, position_array, color_array, normal_position
        )

        pnts = Pnts(PntsHeader(), pnts_body)
        pnts.sync()

        return pnts

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> Pnts:
        """
        Creates a Pnts from an array
        """

        # build tile header
        h_arr = array[0 : PntsHeader.BYTE_LENGTH]
        pnts_header = PntsHeader.from_array(h_arr)

        if pnts_header.tile_byte_length != len(array):
            raise InvalidPntsError(
                f"Invalid byte length in header, the size of array is {len(array)}, "
                f"the tile_byte_length for header is {pnts_header.tile_byte_length}"
            )

        # build tile body
        b_len = (
            pnts_header.ft_json_byte_length
            + pnts_header.ft_bin_byte_length
            + pnts_header.bt_json_byte_length
            + pnts_header.bt_bin_byte_length
        )
        b_arr = array[PntsHeader.BYTE_LENGTH : PntsHeader.BYTE_LENGTH + b_len]
        pnts_body = PntsBody.from_array(pnts_header, b_arr)

        # build the tile with header and body
        return Pnts(pnts_header, pnts_body)

    @staticmethod
    def from_points(
        points: npt.NDArray[np.uint8], include_rgb: bool, include_classification: bool
    ) -> Pnts:
        """
        Create a pnts from an uint8 data array containing:
         - points as SemanticPoint.POSITION
         - if include_rgb, rgb as SemanticPoint.RGB
         - if include_classification, classification as a single np.uint8 value that will put in the batch table
        """
        if len(points) == 0:
            raise ValueError("The argument points cannot be empty.")

        point_size = (
            3 * 4 + (3 if include_rgb else 0) + (1 if include_classification else 0)
        )

        if len(points) % point_size != 0:
            raise ValueError(
                f"The length of points array is {len(points)} but the point size is {point_size}."
                f"There is a rest of {len(points) % point_size}"
            )

        count = len(points) // point_size

        ft = PntsFeatureTable()
        ft.header = PntsFeatureTableHeader.from_semantic(
            SemanticPoint.POSITION,
            SemanticPoint.RGB if include_rgb else None,
            None,
            count,
        )
        ft.body = PntsFeatureTableBody.from_array(ft.header, points)

        bt = BatchTable()
        if include_classification:
            sdt = np.dtype([("Classification", "u1")])
            offset = count * (3 * 4 + (3 if include_rgb else 0))
            bt.add_property_as_binary(
                "Classification",
                points[offset : offset + count * sdt.itemsize],
                "UNSIGNED_BYTE",
                "SCALAR",
            )

        body = PntsBody()
        body.feature_table = ft
        body.batch_table = bt

        pnts = Pnts(PntsHeader(), body)
        pnts.sync()

        return pnts

    @staticmethod
    def from_file(tile_path: Path) -> Pnts:
        with tile_path.open("rb") as f:
            data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return Pnts.from_array(arr)


class PntsHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self) -> None:
        super().__init__()
        self.magic_value = b"pnts"
        self.version = 1

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the header as a numpy array.
        """
        header_arr = np.frombuffer(self.magic_value, np.uint8)

        header_arr2 = np.array(
            [
                self.version,
                self.tile_byte_length,
                self.ft_json_byte_length,
                self.ft_bin_byte_length,
                self.bt_json_byte_length,
                self.bt_bin_byte_length,
            ],
            dtype=np.uint32,
        )

        return np.concatenate((header_arr, header_arr2.view(np.uint8)))

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> PntsHeader:
        """
        Create a PntsHeader from an array
        """

        h = PntsHeader()

        if len(array) != PntsHeader.BYTE_LENGTH:
            raise InvalidPntsError(
                f"Invalid header byte length, the size of array is {len(array)}, "
                f"the header must have a size of {PntsHeader.BYTE_LENGTH}"
            )

        h.version = struct.unpack("i", array[4:8].tobytes())[0]
        h.tile_byte_length = struct.unpack("i", array[8:12].tobytes())[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16].tobytes())[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20].tobytes())[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24].tobytes())[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28].tobytes())[0]

        return h


class PntsBody(TileContentBody):
    def __init__(self) -> None:
        self.feature_table: PntsFeatureTable = PntsFeatureTable()
        self.batch_table = BatchTable()

    def to_array(self) -> npt.NDArray[np.uint8]:
        """
        Returns the body as a numpy array.
        """
        feature_table_array = self.feature_table.to_array()
        batch_table_array = self.batch_table.to_array()
        return np.concatenate((feature_table_array, batch_table_array))

    def get_points(
        self, transform: npt.NDArray[np.float64] | None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8 | np.uint16] | None]:
        fth = self.feature_table.header

        xyz = self.feature_table.body.position.view(np.float32).reshape(
            (fth.points_length, 3)
        )
        if fth.colors == SemanticPoint.RGB:
            rgb = self.feature_table.body.color
            if rgb is None:
                raise InvalidPntsError(
                    "If fth.colors is SemanticPoint.RGB, rgb cannot be None."
                )
            rgb = rgb.reshape((fth.points_length, 3))
        else:
            rgb = None

        if transform is not None:
            transform = transform.reshape((4, 4))
            xyzw = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
            xyz = np.dot(xyzw, transform.astype(xyz.dtype))[:, :3]

        return xyz, rgb

    @staticmethod
    def from_array(header: PntsHeader, array: npt.NDArray[np.uint8]) -> PntsBody:
        """
        Creates a PntsBody from an array and the header
        """

        # build feature table
        feature_table_size = header.ft_json_byte_length + header.ft_bin_byte_length
        feature_table_array = array[:feature_table_size]
        feature_table = PntsFeatureTable.from_array(header, feature_table_array)

        # build batch table
        batch_table_size = header.bt_json_byte_length + header.bt_bin_byte_length
        batch_table_array = array[
            feature_table_size : feature_table_size + batch_table_size
        ]
        batch_table = BatchTable.from_array(
            header, batch_table_array, feature_table.nb_points()
        )

        # build tile body with feature table
        body = PntsBody()
        body.feature_table = feature_table
        body.batch_table = batch_table

        return body
