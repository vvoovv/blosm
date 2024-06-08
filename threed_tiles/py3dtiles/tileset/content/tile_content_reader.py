from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from py3dtiles.exceptions import Invalid3dtilesError

from .b3dm import B3dm
from .pnts import Pnts

if TYPE_CHECKING:
    from py3dtiles.tileset.content import TileContent

__all__ = ["read_binary_tile_content"]


def read_binary_tile_content(tile_path: Path) -> TileContent:
    with tile_path.open("rb") as f:
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)

        tile_content = read_array(arr)
        if tile_content is None or tile_content.header is None:
            raise Invalid3dtilesError(
                f"The file {tile_path} doesn't contain a valid TileContent data."
            )

        return tile_content


def read_array(array: npt.NDArray[np.uint8]) -> TileContent | None:
    magic = "".join([c.decode("UTF-8") for c in array[0:4].view("c")])
    if magic == "pnts":
        return Pnts.from_array(array)
    if magic == "b3dm":
        return B3dm.from_array(array)
    return None
