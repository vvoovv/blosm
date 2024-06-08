from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import Invalid3dtilesError
from py3dtiles.tileset.content.feature_table import (
    FeatureTable,
    FeatureTableBody,
    FeatureTableHeader,
)

if TYPE_CHECKING:
    from .tile_content import TileContentHeader

COMPONENT_TYPE_NUMPY_MAPPING = {
    "BYTE": np.int8,
    "UNSIGNED_BYTE": np.uint8,
    "SHORT": np.int16,
    "UNSIGNED_SHORT": np.uint16,
    "INT": np.int32,
    "UNSIGNED_INT": np.uint32,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}

TYPE_LENGTH_MAPPING = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


ComponentLiteralType = Literal[
    "BYTE",
    "UNSIGNED_BYTE",
    "SHORT",
    "UNSIGNED_SHORT",
    "INT",
    "UNSIGNED_INT",
    "FLOAT",
    "DOUBLE",
]

ComponentNumpyType = Union[
    np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64
]

PropertyLiteralType = Literal["SCALAR", "VEC2", "VEC3", "VEC4"]


class B3dmFeatureTableHeader(FeatureTableHeader):
    def __init__(
        self, data: dict[str, list[Any] | dict[str, Any]] | None = None
    ) -> None:
        if data is not None:
            self.data = data
        else:
            self.data = {}

    def to_array(self) -> npt.NDArray[np.uint8]:
        if not self.data:
            return np.empty((0,), dtype=np.uint8)

        json_str = json.dumps(self.data, separators=(",", ":"))
        if (len(json_str) + 28) % 8 != 0:
            json_str += " " * (8 - (len(json_str) + 28) % 8)
        return np.frombuffer(json_str.encode("utf-8"), dtype=np.uint8)


class B3dmFeatureTableBody(FeatureTableBody):
    def __init__(self, data: list[npt.NDArray[ComponentNumpyType]] | None = None):
        if data is not None:
            self.data = data
        else:
            self.data = []

    def to_array(self) -> npt.NDArray[np.uint8]:
        if not self.data:
            return np.empty((0,), dtype=np.uint8)

        if self.nbytes % 8 != 0:
            padding_str = " " * (8 - self.nbytes % 8)
            padding = np.frombuffer(padding_str.encode("utf-8"), dtype=np.uint8)
            self.data.append(padding)

        return np.concatenate(
            [data.view(np.uint8) for data in self.data], dtype=np.uint8
        )


class B3dmFeatureTable(FeatureTable[B3dmFeatureTableHeader, B3dmFeatureTableBody]):
    """
    Only the JSON header has been implemented for now. According to the feature
    table documentation, the binary body is useful for storing long arrays of
    data (better performances)
    """

    def __init__(self) -> None:
        self.header = B3dmFeatureTableHeader()
        self.body = B3dmFeatureTableBody()

    def get_batch_length(self) -> Any | int:
        return self.header.data.get("BATCH_LENGTH", 0)

    def set_batch_length(self, batch_length: Any) -> None:
        self.header.data["BATCH_LENGTH"] = batch_length

    def add_property_as_json(self, property_name: str, array: list[Any]) -> None:
        self.header.data[property_name] = array

    def add_property_as_binary(
        self,
        property_name: str,
        array: npt.NDArray[ComponentNumpyType],
        component_type: ComponentLiteralType,
        property_type: PropertyLiteralType,
    ) -> None:
        if array.dtype != COMPONENT_TYPE_NUMPY_MAPPING[component_type]:
            raise Invalid3dtilesError(
                "The dtype of array should be the same as component_type,"
                f"the dtype of the array is {array.dtype} and"
                f"the dytpe of {component_type} is {COMPONENT_TYPE_NUMPY_MAPPING[component_type]}"
            )

        self.header.data[property_name] = {
            "byteOffset": self.body.nbytes,
            "componentType": component_type,
            "type": property_type,
        }

        transformed_array = array.reshape(-1)
        self.body.data.append(transformed_array)

    def get_binary_property(
        self, property_name_to_fetch: str
    ) -> npt.NDArray[ComponentNumpyType]:
        binary_property_index = 0
        # The order in self.header.data is the same as in self.body.data
        # We should filter properties added as json.
        for property_name, property_definition in self.header.data.items():
            if isinstance(
                property_definition, list
            ):  # If it is a list, it means that it is a json property
                continue
            elif property_name_to_fetch == property_name:
                return self.body.data[binary_property_index]
            else:
                binary_property_index += 1
        else:
            raise ValueError(f"The property {property_name_to_fetch} is not found")

    def to_array(self) -> npt.NDArray[np.uint8]:
        feature_table_header_array = self.header.to_array()
        feature_table_body_array = self.body.to_array()

        return np.concatenate((feature_table_header_array, feature_table_body_array))

    @staticmethod
    def from_array(
        tile_header: TileContentHeader,
        array: npt.NDArray[np.uint8],
        feature_len: int = 0,
    ) -> B3dmFeatureTable:
        feature_table = B3dmFeatureTable()
        # separate feature table header
        feature_table_header_length = tile_header.ft_json_byte_length
        feature_table_body_array = array[feature_table_header_length:]
        feature_table_header_array = array[0:feature_table_header_length]

        feature_table.header.data = json.loads(
            feature_table_header_array.tobytes().decode("utf-8") or "{}"
        )

        previous_byte_offset = 0
        for property_definition in feature_table.header.data.values():
            if not isinstance(property_definition, dict):
                continue

            if previous_byte_offset != property_definition["byteOffset"]:
                raise ValueError(
                    f"The byte offset is {property_definition['byteOffset']} but the byte offset computed is {previous_byte_offset}"
                )

            numpy_type = COMPONENT_TYPE_NUMPY_MAPPING[
                property_definition["componentType"]
            ]
            end_byte_offset = property_definition["byteOffset"] + (
                np.dtype(numpy_type).itemsize
                * TYPE_LENGTH_MAPPING[property_definition["type"]]
                * feature_len
            )
            feature_table.body.data.append(
                feature_table_body_array[
                    property_definition["byteOffset"] : end_byte_offset
                ].view(numpy_type)
            )
            previous_byte_offset = end_byte_offset

        return feature_table
