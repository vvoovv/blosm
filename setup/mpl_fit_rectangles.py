import os, sys
from building.manager import BaseBuildingManager


def _checkPath():
    path = os.path.abspath(
        os.path.join( os.path.dirname(__file__), os.pardir, os.pardir, "fit_rectangles")
    )
    if path in sys.path:
        sys.path.remove(path)
    # make <path> the first one to search for a module
    sys.path.insert(0, path)
_checkPath()


from mpl.renderer.fit_rectangles import FitRectanglesRenderer


def setup(app, data):
    buildings = BaseBuildingManager(data, app, None, None)
    
    data.addCondition(
        lambda tags, e: "building" in tags,
        "buildings",
        buildings
    )
    
    br = FitRectanglesRenderer()
    
    # <br> stands for "building renderer"
    buildings.setRenderer(br)