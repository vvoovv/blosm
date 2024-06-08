from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .tile_content import TileContentHeader

ComponentNumpyType = Union[
    np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64
]


class FeatureTableHeader:
    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        ...


class FeatureTableBody:
    def __init__(self) -> None:
        self.data: list[npt.NDArray[ComponentNumpyType]] = []

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        ...

    @property
    def nbytes(self) -> int:
        return sum([data.nbytes for data in self.data])


_FeatureTableHeaderT = TypeVar("_FeatureTableHeaderT", bound=FeatureTableHeader)
_FeatureTableBodyT = TypeVar("_FeatureTableBodyT", bound=FeatureTableBody)


class FeatureTable(Generic[_FeatureTableHeaderT, _FeatureTableBodyT]):
    """
    Only the JSON header has been implemented for now. According to the feature
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    header: _FeatureTableHeaderT
    body: _FeatureTableBodyT

    @abstractmethod
    def to_array(self) -> npt.NDArray[np.uint8]:
        ...

    @staticmethod
    @abstractmethod
    def from_array(
        tile_header: TileContentHeader, array: npt.NDArray[np.uint8]
    ) -> FeatureTable[Any, Any]:
        ...
