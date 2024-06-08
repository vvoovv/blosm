from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import InvalidTilesetError, Py3dtilesException, TilerException
from py3dtiles.typing import RefineType, TileDictType

from .bounding_volume import BoundingVolume
from .bounding_volume_box import BoundingVolumeBox
from .content import TileContent, read_binary_tile_content
from .root_property import RootProperty

if TYPE_CHECKING:
    from py3dtiles.tileset import TileSet

DEFAULT_TRANSFORMATION = np.identity(4, dtype=np.float64)
DEFAULT_TRANSFORMATION.setflags(write=False)


class Tile(RootProperty[TileDictType]):
    def __init__(
        self,
        geometric_error: float = 500,
        bounding_volume: BoundingVolume[Any] | None = None,
        transform: npt.NDArray[np.float64] = DEFAULT_TRANSFORMATION,
        refine_mode: RefineType = "ADD",
        content_uri: Path | None = None,
    ) -> None:
        super().__init__()
        self.bounding_volume = bounding_volume
        self.geometric_error = geometric_error
        self._refine: RefineType = "ADD"
        self.set_refine_mode(refine_mode)
        self.tile_content: TileContent | TileSet | None = None
        self.content_uri: Path | None = content_uri
        self.children: list[Tile] = []
        # Some possible valid properties left un-delt with viewerRequestVolume
        self.transform = transform

    @classmethod
    def from_dict(cls, tile_dict: TileDictType) -> Tile:
        if "box" in tile_dict["boundingVolume"]:
            bounding_volume = BoundingVolumeBox.from_dict(tile_dict["boundingVolume"])  # type: ignore [arg-type]
        elif (
            "region" in tile_dict["boundingVolume"]
            or "sphere" in tile_dict["boundingVolume"]
        ):
            raise NotImplementedError(
                "The support of bounding volume region and sphere is not implemented yet"
            )
        else:
            raise InvalidTilesetError(
                f"The bounding volume {list(tile_dict['boundingVolume'].keys())[0]} is unknown"
            )

        tile = cls(
            geometric_error=tile_dict["geometricError"],
            bounding_volume=bounding_volume,
        )

        if "refine" in tile_dict:
            tile.set_refine_mode(tile_dict["refine"])

        if "transform" in tile_dict:
            tile.transform = np.array(tile_dict["transform"]).reshape((4, 4))

        if "children" in tile_dict:
            for child in tile_dict["children"]:
                tile.children.append(Tile.from_dict(child))

        if "content" in tile_dict:
            tile.content_uri = Path(tile_dict["content"]["uri"])

        tile.set_properties_from_dict(tile_dict)

        return tile

    def get_or_fetch_content(self, root_uri: Path | None) -> TileContent | TileSet:
        """
        If a `tile_content` content has been set, returns this content.
        If the tile content is None and a `tile_content` uri has been set, the tile will load the file and return its content.

        :param root_uri: the base uri which `tile.content_uri` is relative to. Usually the directory containing the tileset containing this tile.
        """
        self._load_tile_content(root_uri)
        if self.tile_content is None:
            raise Py3dtilesException(
                "self.tile_content cannot be None here, seems to be a py3dtiles issue."
            )
        return self.tile_content

    def has_content(self) -> bool:
        """
        Returns if there is a tile content (loaded or not).
        """
        return bool(self.tile_content is not None or self.content_uri)

    def has_content_loaded(self) -> bool:
        """
        Returns if there is a tile content loaded in this tile.
        """
        return self.tile_content is not None

    def delete_on_disk(self, root_uri: Path, delete_sub_tileset: bool = False) -> None:
        """
        Deletes all files linked to the tile and its children. The uri of the folder where tileset is, should be defined.

        :param root_uri: The folder where tileset is
        :param delete_sub_tileset: If True, all tilesets present as tile content will be removed as well as their content.
        If False, the linked tilesets in tiles won't be removed.
        """
        for child in self.children:
            child.delete_on_disk(root_uri, delete_sub_tileset)

        # if there is no content_uri, there is no file to remove
        if self.content_uri is None:
            return

        if self.content_uri.is_absolute():
            tile_content_path = self.content_uri
        else:
            tile_content_path = root_uri / self.content_uri

        if tile_content_path.suffix == ".json":
            if delete_sub_tileset:
                self.get_or_fetch_content(root_uri).delete_on_disk(tile_content_path)  # type: ignore
        else:
            tile_content_path.unlink()

    def set_refine_mode(self, mode: RefineType) -> None:
        if mode != "ADD" and mode != "REPLACE":
            raise InvalidTilesetError(
                f"Unknown refinement mode {mode}. Should be either 'ADD' or 'REPLACE'."
            )
        self._refine = mode

    def get_refine_mode(self) -> RefineType:
        return self._refine

    def add_child(self, tile: Tile) -> None:
        self.children.append(tile)

        if tile.bounding_volume is not None:
            if self.bounding_volume is None:
                self.bounding_volume = copy.deepcopy(tile.bounding_volume)
                self.bounding_volume.transform(tile.transform)
            else:
                transformed_bounding_volume = copy.deepcopy(tile.bounding_volume)
                transformed_bounding_volume.transform(tile.transform)
                parent_inv_transform = np.linalg.inv(self.transform)
                transformed_bounding_volume.transform(parent_inv_transform)
                self.bounding_volume.add(transformed_bounding_volume)

    def get_all_children(self) -> list[Tile]:
        """
        :return: the recursive (across the children tree) list of the children
                 tiles
        """
        descendants = []
        for child in self.children:
            # Add the child...
            descendants.append(child)
            # and if (and only if) they are grand-children then recurse
            if child.children:
                descendants += child.get_all_children()
        return descendants

    def sync_bounding_volume_with_children(self) -> None:
        if self.bounding_volume is None:
            raise TilerException("This Tile has no bounding volume.")
        if not self.bounding_volume.is_box():
            raise NotImplementedError(
                "Don't know yet how to sync non box bounding volume."
            )

        # We consider that whatever information is present it is the
        # proper one (in other terms: when they are no sub-tiles this tile
        # is a leaf-tile and thus is has no synchronization to do)
        for child in self.children:
            child.sync_bounding_volume_with_children()

        # The information that depends on (is defined by) the children
        # nodes is limited to be bounding volume.
        self.bounding_volume.sync_with_children(self)

    def write_content(self, root_uri: Path | None) -> None:
        """
        Write (or overwrite) the tile *content* to the directory specified
        as parameter and withing the relative filename designated by
        the tile's content uri.

        :param root_uri: the base uri of this tile, usually the folder where the tileset is
        """

        if self.tile_content is None:
            raise TilerException(
                "The tile has no tile content. "
                "A tile content should be added in the tile."
            )

        if self.content_uri is None:
            raise TilerException("tile.content_uri is None, cannot write tile content")

        if self.content_uri.is_absolute():
            content_path = self.content_uri
        else:
            if root_uri is None:
                raise ValueError(
                    "No root_uri given and tile.content_uri is not absolute"
                )
            content_path = root_uri / self.content_uri

        # Make sure the output directory exists (note that target_dir may
        # be a sub-directory of 'directory' because the uri might hold its
        # own path):
        content_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self.tile_content, TileContent):
            self.tile_content.save_as(content_path)
        else:
            self.tile_content.write_to_directory(content_path)

    def to_dict(self) -> TileDictType:
        if self.bounding_volume is not None:
            bounding_volume = self.bounding_volume
        else:
            raise InvalidTilesetError("Bounding volume is not set")
        bounding_volume_dict = bounding_volume.to_dict()

        refine = self._refine
        if refine not in ["ADD", "REPLACE"]:
            raise InvalidTilesetError(
                f"refine should be either ADD or REPLACE, currently {refine}."
            )

        # Mandatory items
        dict_data: TileDictType = {
            "boundingVolume": bounding_volume_dict,
            "geometricError": self.geometric_error,
            "refine": refine,
        }

        dict_data = self.add_root_properties_to_dict(dict_data)

        if (
            self.transform is not None and self.transform is not DEFAULT_TRANSFORMATION
        ):  # if transform has not the same id
            dict_data["transform"] = list(self.transform.flatten())

        if self.children:
            # The children list exists indeed (for technical reasons) yet it
            # happens to be still empty. This would pollute the json output
            # by adding a "children" entry followed by an empty list. In such
            # case just remove that attributes entry:
            dict_data["children"] = [child.to_dict() for child in self.children]

        if self.content_uri:
            dict_data["content"] = {"uri": str(self.content_uri)}
        return dict_data

    def _load_tile_content(self, root_uri: Path | None) -> None:
        if self.tile_content:
            return

        if not self.content_uri:
            raise RuntimeError("Cannot load a tile without a content_uri")

        if self.content_uri.is_absolute():
            uri = self.content_uri
        else:
            if root_uri is None:
                raise RuntimeError(
                    "Cannot load a tile without a root_uri if self.content_uri is relative"
                )
            uri = root_uri / self.content_uri

        if uri.suffix == ".json":
            from .tileset import TileSet

            self.tile_content = TileSet.from_file(uri)
        else:
            self.tile_content = read_binary_tile_content(uri)
