from __future__ import annotations

from abc import ABC, abstractmethod

from py3dtiles.typing import ExtensionDictType


class BaseExtension(ABC):
    """
    A base class to manage 3dtiles extension.

    If an extension is added somewhere in a tileset,
    the user must add the name of the extension in the attribute `extensions_used` of the class `TileSet`.
    Also, if the support of an extension is necessary to display the tileset,
    the name must be added in the attribute `extensions_required` of the class `TileSet`.
    """

    def __init__(self, name: str):
        self.name = name

    @classmethod
    @abstractmethod
    def from_dict(cls, extension_dict: ExtensionDictType) -> BaseExtension:
        ...

    @abstractmethod
    def to_dict(self) -> ExtensionDictType:
        ...
