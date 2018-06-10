from setup.premium import setup_base

def setup(app, osm):
    setup_base(app, osm, getMaterials, bldgPreRender)


from realistic.material.renderer import\
    SeamlessTexture, SeamlessTextureWithColor, MaterialWithColor,\
    SeamlessTextureScaledWithColor, SeamlessTextureScaled, FacadeSeamlessTexture, FacadeWithOverlay
from realistic.material.colors import brickColors, plasterColors, glassColors


def getMaterials():
    return dict(
        #neoclassical = (FacadeWithOverlay, "plaster", plasterColors),
        glass = (FacadeSeamlessTexture, glassColors),
        commercial = (FacadeWithOverlay, "brick", brickColors),
        residential = (FacadeWithOverlay, "plaster", plasterColors),
        brick = SeamlessTexture,
        brick_color = SeamlessTextureWithColor,
        concrete = SeamlessTexture,
        concrete_color = SeamlessTextureWithColor,
        plaster = SeamlessTexture,
        plaster_color = SeamlessTextureWithColor,
        roof_tiles = SeamlessTexture,
        roof_tiles_color = SeamlessTextureWithColor,
        metal_without_uv = MaterialWithColor,
        metal = SeamlessTexture,
        metal_color = SeamlessTextureWithColor,
        metal_scaled = SeamlessTextureScaled,
        metal_scaled_color = SeamlessTextureScaledWithColor
    )


def bldgPreRender(building):
    element = building.element
    tags = element.tags
    
    # material for walls
    material = building.wallsMaterial
    if not material in ("plaster", "brick", "metal", "glass", "mirror"):
        material = "plaster"
    # tb stands for "OSM tag building"
    tb = building.getOsmTagValue("building")
    
    if tb in ("cathedral", "wall") or\
        building.getOsmTagValue("amenity") == "place_of_worship" or\
        building.getOsmTagValue("man_made") == "tower":
        building.setMaterialWalls(material, addColorSuffix=True)
    else:
        if material == "glass" or material == "mirror":
            building.setMaterialWalls("glass")
        elif tb in ("residential", "apartments"):
            building.setMaterialWalls("residential")
        else:
            building.setMaterialWalls("commercial")
    
    # material for roof
    material = building.roofMaterial
    if not material in ("metal", "concrete", "roof_tiles"):
        material = "metal"
    
    if material == "concrete":
        building.setMaterialRoof("concrete")
    elif material == "roof_tiles":
        building.setMaterialRoof("roof_tiles")
    elif material == "metal":
        roofShape = tags.get("roof:shape")
        if roofShape == "onion":
            building.setMaterialRoof("metal_without_uv", addColorSuffix=False)
        elif roofShape == "dome":
            building.setMaterialRoof("metal_scaled")
        else:
            building.setMaterialRoof("metal")