from mpl.renderer.roof_profile import RoofProfileRenderer
from building.manager import BaseBuildingManager


def setup(app, data):
    buildings = BaseBuildingManager(data, app, None, None)
    
    data.addCondition(
        lambda tags, e: "building" in tags,
        "buildings",
        buildings
    )
    
    br = RoofProfileRenderer(app, data)
    
    # <br> stands for "building renderer"
    buildings.setRenderer(br)