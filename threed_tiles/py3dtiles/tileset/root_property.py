from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from py3dtiles.typing import ExtraDictType, RootPropertyDictType

if TYPE_CHECKING:
    from typing_extensions import Self

    from py3dtiles.tileset.extension import BaseExtension

_JsonDictT = TypeVar("_JsonDictT", bound=RootPropertyDictType)


class RootProperty(ABC, Generic[_JsonDictT]):
    """
    One the 3DTiles notions defined as an abstract data model through
    a schema of the 3DTiles specifications (either core of extensions).
    """

    def __init__(self) -> None:
        self.extensions: dict[str, BaseExtension] = {}
        self.extras: ExtraDictType = {}

    @classmethod
    @abstractmethod
    def from_dict(cls, data_dict: _JsonDictT) -> Self:
        ...

    @abstractmethod
    def to_dict(self) -> _JsonDictT:
        ...

    def add_root_properties_to_dict(self, dict_data: _JsonDictT) -> _JsonDictT:
        # we cannot merge root_property_data without mypy issues
        if self.extensions:
            dict_data["extensions"] = {
                name: extension.to_dict() for name, extension in self.extensions.items()
            }

        if self.extras:
            dict_data["extras"] = self.extras

        return dict_data

    def set_properties_from_dict(
        self,
        dict_data: _JsonDictT,
    ) -> None:
        self.extensions = {}  # TODO not yet implemented
        if "extras" in dict_data:
            self.extras = dict_data["extras"]
