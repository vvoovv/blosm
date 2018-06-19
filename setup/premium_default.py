from setup.premium import setup_base

def setup(app, osm):
    setup_base(app, osm, getMaterials, bldgPreRender)


from realistic.material.renderer import\
    SeamlessTexture, SeamlessTextureWithColor, MaterialWithColor,\
    SeamlessTextureScaledWithColor, SeamlessTextureScaled, FacadeSeamlessTexture, FacadeWithOverlay
from realistic.material.colors import\
    brickColors, plasterColors, glassColors, concreteColors, roofTilesColors, metalColors


def getMaterials():
    return dict(
        #neoclassical = (FacadeWithOverlay, "plaster", plasterColors),
        glass = (FacadeSeamlessTexture, glassColors),
        commercial = (FacadeWithOverlay, "brick", brickColors),
        residential = (FacadeWithOverlay, "plaster", plasterColors),
        brick = SeamlessTexture,
        brick_color = (SeamlessTextureWithColor, brickColors),
        concrete = SeamlessTexture,
        concrete_color = (SeamlessTextureWithColor, concreteColors),
        plaster = SeamlessTexture,
        plaster_color = (SeamlessTextureWithColor, plasterColors),
        roof_tiles = SeamlessTexture,
        roof_tiles_color = (SeamlessTextureWithColor, roofTilesColors),
        metal = SeamlessTexture,
        metal_color = (SeamlessTextureWithColor, metalColors),
        metal_without_uv = MaterialWithColor,
        metal_scaled = SeamlessTextureScaled,
        metal_scaled_color = (SeamlessTextureScaledWithColor, metalColors)
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
        building.setMaterialWalls("%s_color" % material)
    else:
        if material == "glass" or material == "mirror":
            building.setMaterialWalls("glass")
        elif tb in ("residential", "apartments"):
            building.setMaterialWalls("residential")
        else:
            building.setMaterialWalls("commercial")
    
    # material for roof
    material = building.roofMaterial  
    roofShape = tags.get("roof:shape")
    
    if material == "concrete" or roofShape == "flat":
        building.setMaterialRoof("concrete_color")
    elif material == "roof_tiles":
        building.setMaterialRoof("roof_tiles_color")
    elif material == "metal":
        if roofShape == "onion":
            building.setMaterialRoof("metal_without_uv")
        elif roofShape == "dome":
            building.setMaterialRoof("metal_scaled_color")
        else:
            building.setMaterialRoof("metal_color")