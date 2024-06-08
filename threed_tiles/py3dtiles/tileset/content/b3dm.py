from __future__ import annotations

import struct

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidB3dmError

from .b3dm_feature_table import B3dmFeatureTable
from .batch_table import BatchTable
from .gltf import GlTF
from .tile_content import TileContent, TileContentBody, TileContentHeader


class B3dm(TileContent):
    def __init__(self, header: B3dmHeader, body: B3dmBody) -> None:
        super().__init__()

        self.header: B3dmHeader = header
        self.body: B3dmBody = body

    def sync(self) -> None:
        """
        Allow to synchronize headers with contents.
        """

        # extract array
        gltf_arr = self.body.gltf.to_array()

        # sync the tile header with feature table contents
        self.header.tile_byte_length = len(gltf_arr) + B3dmHeader.BYTE_LENGTH
        self.header.bt_json_byte_length = 0
        self.header.bt_bin_byte_length = 0
        self.header.ft_json_byte_length = 0
        self.header.ft_bin_byte_length = 0

        if self.body.feature_table is not None:
            fth_arr = self.body.feature_table.to_array()

            self.header.tile_byte_length += len(fth_arr)
            self.header.ft_json_byte_length = len(fth_arr)

        if self.body.batch_table is not None:
            bth_arr = self.body.batch_table.to_array()

            self.header.tile_byte_length += len(bth_arr)
            self.header.bt_json_byte_length = len(bth_arr)

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
            print("Batch table json byte length: ", th.bt_json_byte_length)
            print("Batch table bin byte length: ", th.bt_bin_byte_length)
        else:
            print("Tile with no header")

        if self.body:
            gltf_header = self.body.gltf.header
            print("")
            print("glTF Header")
            print("-----------")
            print(gltf_header)
        else:
            print("Tile with no body")

    @staticmethod
    def from_gltf(gltf: GlTF, batch_table: BatchTable | None = None) -> B3dm:
        b3dm_body = B3dmBody()
        b3dm_body.gltf = gltf
        if batch_table is not None:
            b3dm_body.batch_table = batch_table

        b3dm_header = B3dmHeader()
        b3dm = B3dm(b3dm_header, b3dm_body)
        b3dm.sync()

        return b3dm

    @staticmethod
    def from_array(array: npt.NDArray[np.uint8]) -> B3dm:
        # build tile header
        h_arr = array[0 : B3dmHeader.BYTE_LENGTH]
        b3dm_header = B3dmHeader.from_array(h_arr)

        if b3dm_header.tile_byte_length != len(array):
            raise InvalidB3dmError(
                f"Invalid byte length in header, the size of array is {len(array)}, "
                f"the tile_byte_length for header is {b3dm_header.tile_byte_length}"
            )

        # build tile body
        b_arr = array[B3dmHeader.BYTE_LENGTH : b3dm_header.tile_byte_length]
        b3dm_body = B3dmBody.from_array(b3dm_header, b_arr)

        # build tile with header and body
        return B3dm(b3dm_header, b3dm_body)


class B3dmHeader(TileContentHeader):
    BYTE_LENGTH = 28

    def __init__(self) -> None:
        super().__init__()
        self.magic_value = b"b3dm"
        self.version = 1

    def to_array(self) -> npt.NDArray[np.uint8]:
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
    def from_array(array: npt.NDArray[np.uint8]) -> B3dmHeader:
        h = B3dmHeader()

        if len(array) != B3dmHeader.BYTE_LENGTH:
            raise InvalidB3dmError(
                f"Invalid header byte length, the size of array is {len(array)}, "
                f"the header must have a size of {B3dmHeader.BYTE_LENGTH}"
            )

        h.version = struct.unpack("i", array[4:8].tobytes())[0]
        h.tile_byte_length = struct.unpack("i", array[8:12].tobytes())[0]
        h.ft_json_byte_length = struct.unpack("i", array[12:16].tobytes())[0]
        h.ft_bin_byte_length = struct.unpack("i", array[16:20].tobytes())[0]
        h.bt_json_byte_length = struct.unpack("i", array[20:24].tobytes())[0]
        h.bt_bin_byte_length = struct.unpack("i", array[24:28].tobytes())[0]

        return h


class B3dmBody(TileContentBody):
    def __init__(self) -> None:
        self.batch_table = BatchTable()
        self.feature_table: B3dmFeatureTable = B3dmFeatureTable()
        self.gltf = GlTF()

    def to_array(self) -> npt.NDArray[np.uint8]:
        if self.feature_table:
            feature_table = self.feature_table.to_array()
        else:
            feature_table = np.array([], dtype=np.uint8)

        if self.batch_table:
            batch_table = self.batch_table.to_array()
        else:
            batch_table = np.array([], dtype=np.uint8)

        # The glTF part must start and end on an 8-byte boundary
        return np.concatenate((feature_table, batch_table, self.gltf.to_array()))

    @staticmethod
    def from_gltf(gltf: GlTF) -> B3dmBody:
        # build tile body
        b = B3dmBody()
        b.gltf = gltf

        return b

    @staticmethod
    def from_array(b3dm_header: B3dmHeader, array: npt.NDArray[np.uint8]) -> B3dmBody:
        # build feature table
        ft_len = b3dm_header.ft_json_byte_length + b3dm_header.ft_bin_byte_length

        # build batch table
        bt_len = b3dm_header.bt_json_byte_length + b3dm_header.bt_bin_byte_length

        # build glTF
        gltf_len = (
            b3dm_header.tile_byte_length - ft_len - bt_len - B3dmHeader.BYTE_LENGTH
        )
        gltf_arr = array[ft_len + bt_len : ft_len + bt_len + gltf_len]
        gltf = GlTF.from_array(gltf_arr)

        # build tile body with batch table
        b = B3dmBody()
        b.gltf = gltf
        if ft_len > 0:
            b.feature_table = B3dmFeatureTable.from_array(b3dm_header, array[:ft_len])
        if bt_len > 0:
            batch_len = b.feature_table.get_batch_length()
            b.batch_table = BatchTable.from_array(
                b3dm_header, array[ft_len : ft_len + bt_len], batch_len
            )

        return b
