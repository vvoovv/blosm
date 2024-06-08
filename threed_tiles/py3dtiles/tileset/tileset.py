from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal

from py3dtiles.typing import AssetDictType, GeometricErrorType, TilesetDictType

from .root_property import RootProperty
from .tile import Tile

if TYPE_CHECKING:
    from typing_extensions import Self

    from .content import TileContent


class Asset(RootProperty[AssetDictType]):
    def __init__(
        self, version: Literal["1.0", "1.1"] = "1.0", tileset_version: str | None = None
    ) -> None:
        super().__init__()
        self.version = version
        self.tileset_version = tileset_version

    @classmethod
    def from_dict(cls, asset_dict: AssetDictType) -> Self:
        asset = cls(asset_dict["version"])
        if "tilesetVersion" in asset_dict:
            asset.tileset_version = asset_dict["tilesetVersion"]

        asset.set_properties_from_dict(asset_dict)

        return asset

    def to_dict(self) -> AssetDictType:
        asset_dict: AssetDictType = {"version": self.version}

        asset_dict = self.add_root_properties_to_dict(asset_dict)

        if self.tileset_version is not None:
            asset_dict["tilesetVersion"] = self.tileset_version

        return asset_dict


class TileSet(RootProperty[TilesetDictType]):
    def __init__(
        self,
        geometric_error: float = 500,
        root_uri: Path | None = None,
    ) -> None:
        super().__init__()
        self.asset = Asset(version="1.0")
        self.geometric_error: GeometricErrorType = geometric_error
        self.root_tile = Tile()
        self.root_uri = root_uri
        self.extensions_used: set[str] = set()
        self.extensions_required: set[str] = set()

    @classmethod
    def from_dict(cls, tileset_dict: TilesetDictType) -> Self:
        tileset = cls(geometric_error=tileset_dict["geometricError"])

        tileset.asset = Asset.from_dict(tileset_dict["asset"])
        tileset.root_tile = Tile.from_dict(tileset_dict["root"])
        tileset.set_properties_from_dict(tileset_dict)

        if "extensionsUsed" in tileset_dict:
            tileset.extensions_used = set(tileset_dict["extensionsUsed"])
        if "extensionsRequired" in tileset_dict:
            tileset.extensions_required = set(tileset_dict["extensionsRequired"])

        return tileset

    @staticmethod
    def from_file(tileset_path: Path) -> TileSet:
        with tileset_path.open() as f:
            tileset_dict = json.load(f)

        tileset = TileSet.from_dict(tileset_dict)
        tileset.root_uri = tileset_path.parent

        return tileset

    def get_all_tile_contents(
        self,
    ) -> Generator[TileContent | TileSet, None, None]:
        tiles = [self.root_tile] + self.root_tile.get_all_children()
        for tile in tiles:
            yield tile.get_or_fetch_content(self.root_uri)

    def delete_on_disk(
        self, tileset_path: Path, delete_sub_tileset: bool = False
    ) -> None:
        """
        Deletes all files linked to the tileset. The uri of the tileset should be defined.

        :param tileset_path: The path of the tileset
        :param delete_sub_tileset: If True, all tilesets present as tile content will be removed as well as their content.
        If False, the linked tilesets in tiles won't be removed.
        """
        tileset_path.unlink()
        self.root_tile.delete_on_disk(tileset_path.parent, delete_sub_tileset)

    def write_to_directory(self, tileset_path: Path, overwrite: bool = False) -> None:
        """
        Write (or overwrite), to the directory whose name is provided, the
        TileSet that is:
        - the tileset as a json file and
        - all the tiles content of the Tiles used by the Tileset.
        :param tileset_path: the target directory name
        :param overwrite: delete the tileset (and the content) if already exists
        """
        if tileset_path.exists():
            if overwrite:
                tileset = TileSet.from_file(tileset_path)
                tileset.delete_on_disk(tileset_path, delete_sub_tileset=True)
            else:
                raise FileExistsError(
                    f"There is already a file at {tileset_path} and overwrite is False"
                )

        # Proceed with the writing of the TileSet per se:
        self.write_as_json(tileset_path)

        # Prior to writing the TileSet, the future location of the enclosed
        # Tile's content (set as their respective TileContent uri) must be
        # specified:
        all_tiles = self.root_tile.get_all_children()
        all_tiles.append(self.root_tile)
        for tile in all_tiles:
            if tile.tile_content is not None:
                tile.write_content(tileset_path.parent)

    def write_as_json(self, tileset_path: Path) -> None:
        """
        Write the tileset as a JSON file.
        :param tileset_path: the path where the tileset will be written
        """
        with tileset_path.open("w") as f:
            f.write(self.to_json())

    def to_dict(self) -> TilesetDictType:
        """
        Convert to json string possibly mentioning used schemas
        """
        # Make sure the TileSet is aligned with its children Tiles.
        self.root_tile.sync_bounding_volume_with_children()

        tileset_dict: TilesetDictType = {
            "root": self.root_tile.to_dict(),
            "asset": self.asset.to_dict(),
            "geometricError": self.geometric_error,
        }

        tileset_dict = self.add_root_properties_to_dict(tileset_dict)

        if self.extensions_used:
            tileset_dict["extensionsUsed"] = list(self.extensions_used)
        if self.extensions_required:
            tileset_dict["extensionsRequired"] = list(self.extensions_required)

        return tileset_dict

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
