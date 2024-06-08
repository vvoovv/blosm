from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .batch_table import BatchTable
from .feature_table import FeatureTable


class TileContent(ABC):
    header: TileContentHeader
    body: TileContentBody

    def to_array(self) -> npt.NDArray[np.uint8]:
        self.sync()
        header_arr = self.header.to_array()
        body_arr = self.body.to_array()
        return np.concatenate((header_arr, body_arr))

    def to_hex_str(self) -> str:
        arr = self.to_array()
        return " ".join(f"{x:02X}" for x in arr)

    def save_as(self, path: Path) -> None:
        tile_arr = self.to_array()
        with path.open("bw") as f:
            f.write(bytes(tile_arr))

    @abstractmethod
    def print_info(self) -> None:
        ...

    @abstractmethod
    def sync(self) -> None:
        """
        Allow to synchronize headers with contents.
        """

    @staticmethod
    @abstractmethod
    def from_array(array: npt.NDArray[np.uint8]) -> TileContent:
        ...


class TileContentHeader(ABC):
    magic_value: Literal[b"b3dm", b"pnts"]
    version: int

    def __init__(self) -> None:
        self.tile_byte_length = 0
        self.ft_json_byte_length = 0
        self.ft_bin_byte_length = 0
        self.bt_json_byte_length = 0
        self.bt_bin_byte_length = 0
        self.bt_length = 0  # number of models in the batch

    @staticmethod
    @abstractmethod
    def from_array(array: npt.NDArray[np.uint8]) -> TileContentHeader:
        ...

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        ...


class TileContentBody(ABC):
    batch_table: BatchTable
    feature_table: FeatureTable[Any, Any]

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        ...
