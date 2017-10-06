from setup.premium import setup_base

def setup(app, osm):
    setup_base(app, osm, getMaterials, bldgPreRender)


from realistic.material.renderer import\
    SeamlessTexture, SeamlessTextureWithColor, MaterialWithColor,\
    SeamlessTextureScaledWithColor, SeamlessTextureScaled, FacadeSeamlessTexture


def getMaterials():
    return dict(
        glass = FacadeSeamlessTexture,
        commercial = FacadeSeamlessTexture,
        apartments = FacadeSeamlessTexture,
        brick = SeamlessTexture,
        brick_with_color = SeamlessTextureWithColor,
        concrete = SeamlessTexture,
        concrete_with_color = SeamlessTextureWithColor,
        plaster = SeamlessTexture,
        plaster_with_color = SeamlessTextureWithColor,
        roof_tiles = SeamlessTexture,
        roof_tiles_with_color = SeamlessTextureWithColor,
        metal_without_uv = MaterialWithColor,
        metal = SeamlessTexture,
        metal_with_color = SeamlessTextureWithColor,
        metal_scaled = SeamlessTextureScaled,
        metal_scaled_with_color = SeamlessTextureScaledWithColor
    )


def bldgPreRender(building):
    element = building.element
    tags = element.tags
    
    # material for walls
    material = building.wallsMaterial
    
    if material == "brick":
        building.setMaterialWalls("brick")
    elif material == "plaster":
        building.setMaterialWalls("plaster")
    elif material == "glass":
        building.setMaterialWalls("glass", False)
    elif tags.get("building") == "commercial":
        building.setMaterialWalls("commercial", False)
    else:
        building.setMaterialWalls("plaster")
    
    # material for roof
    material = building.roofMaterial
    if not material:
        material = "metal"
    
    if material == "concrete":
        building.setMaterialRoof("concrete")
    elif material == "roof_tiles":
        building.setMaterialRoof("roof_tiles")
    elif material == "metal":
        roofShape = tags.get("roof:shape")
        if roofShape == "onion":
            building.setMaterialRoof("metal_without_uv", False)
        elif roofShape == "dome":
            building.setMaterialRoof("metal_scaled")
        else:
            building.setMaterialRoof("metal")