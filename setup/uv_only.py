from setup.premium import setup_base

def setup(app, osm):
    setup_base(app, osm, getMaterials, bldgPreRender)


from realistic.material.renderer import UvOnly


def getMaterials():
    return dict(
        uv_only = UvOnly
    )


def bldgPreRender(building):
    building.setMaterialWalls("uv_only")
    building.setMaterialRoof("uv_only")