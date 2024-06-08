import json
from pathlib import Path

from .content import Pnts, read_binary_tile_content


def number_of_points_in_tileset(tileset_path: Path) -> int:
    with tileset_path.open() as f:
        tileset = json.load(f)

    nb_points = 0

    children_tileset_info = [(tileset["root"], tileset["root"]["refine"])]
    while children_tileset_info:
        child_tileset, parent_refine = children_tileset_info.pop()
        child_refine = (
            child_tileset["refine"] if child_tileset.get("refine") else parent_refine
        )

        if "content" in child_tileset:
            content = tileset_path.parent / child_tileset["content"]["uri"]

            pnts_should_count = "children" not in child_tileset or child_refine == "ADD"
            if content.suffix == ".pnts" and pnts_should_count:
                tile = read_binary_tile_content(content)
                if isinstance(tile, Pnts):
                    nb_points += tile.body.feature_table.nb_points()
            elif content.suffix == ".json":
                with content.open() as f:
                    sub_tileset = json.load(f)
                children_tileset_info.append((sub_tileset["root"], child_refine))

        if "children" in child_tileset:
            children_tileset_info += [
                (sub_child_tileset, child_refine)
                for sub_child_tileset in child_tileset["children"]
            ]

    return nb_points
