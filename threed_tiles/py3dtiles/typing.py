from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import numpy.typing as npt
#from pyproj import CRS

if TYPE_CHECKING:
    from typing_extensions import NotRequired

# Tileset types

ExtensionDictType = Dict[str, Any]
ExtraDictType = Dict[str, Any]
GeometricErrorType = float
PropertyType = Dict[str, Any]
RefineType = Literal["ADD", "REPLACE"]
TransformDictType = List[float]


class RootPropertyDictType(TypedDict):
    extensions: NotRequired[dict[str, ExtensionDictType]]
    extras: NotRequired[ExtraDictType]


class BoundingVolumeBoxDictType(RootPropertyDictType):
    box: list[float]


class BoundingVolumeRegionDictType(RootPropertyDictType):
    region: list[float]


class BoundingVolumeSphereDictType(RootPropertyDictType):
    sphere: list[float]


BoundingVolumeDictType = Union[
    BoundingVolumeBoxDictType,
    BoundingVolumeRegionDictType,
    BoundingVolumeSphereDictType,
]


class ContentType(RootPropertyDictType):
    boundingVolume: NotRequired[BoundingVolumeDictType]
    uri: str


class PropertyDictType(RootPropertyDictType):
    maximum: float
    minimum: float


class AssetDictType(RootPropertyDictType):
    version: Literal["1.0", "1.1"]
    tilesetVersion: NotRequired[str]


class TileDictType(RootPropertyDictType):
    boundingVolume: BoundingVolumeDictType
    geometricError: GeometricErrorType
    viewerRequestVolume: NotRequired[BoundingVolumeDictType]
    refine: NotRequired[RefineType]
    transform: NotRequired[TransformDictType]
    content: NotRequired[ContentType]
    children: NotRequired[list[TileDictType]]


class TilesetDictType(RootPropertyDictType):
    asset: AssetDictType
    geometricError: GeometricErrorType
    root: TileDictType
    properties: NotRequired[PropertyType]
    extensionsUsed: NotRequired[list[str]]
    extensionsRequired: NotRequired[list[str]]


# Tile content types

BatchTableHeaderDataType = Dict[str, Union[List[Any], Dict[str, Any]]]

FeatureTableHeaderDataType = Dict[
    str,
    Union[
        int,  # points_length
        Dict[str, int],  # byte offsets
        Tuple[float, float, float],  # rtc
        List[float],  # quantized_volume_offset and quantized_volume_scale
        Tuple[int, int, int, int],  # constant_rgba
    ],
]


class HierarchyClassDictType(TypedDict):
    name: str
    length: int
    instances: dict[str, list[Any]]


# Tiler types

PortionItemType = Tuple[int, ...]
PortionsType = List[Tuple[str, PortionItemType]]


class MetadataReaderType(TypedDict):
    portions: PortionsType
    aabb: npt.NDArray[np.float64]
    crs_in: CRS | None
    point_count: int
    avg_min: npt.NDArray[np.float64]


OffsetScaleType = Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    Optional[npt.NDArray[np.float64]],
    Optional[float],
]
