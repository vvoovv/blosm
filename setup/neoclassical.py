"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
        neoclassical = (FacadeWithOverlay, "plaster", plasterColors),
        glass = (FacadeSeamlessTexture, glassColors),
        office = (FacadeWithOverlay, "brick", brickColors),
        appartments = (FacadeWithOverlay, "plaster", plasterColors),
        #brick = SeamlessTexture,
        brick_color = (SeamlessTextureWithColor, brickColors),
        #concrete = SeamlessTexture,
        concrete_color = (SeamlessTextureWithColor, concreteColors),
        #gravel = SeamlessTexture,
        gravel_color = (SeamlessTextureWithColor, concreteColors),
        #plaster = SeamlessTexture,
        plaster_color = (SeamlessTextureWithColor, plasterColors),
        #roof_tiles = SeamlessTexture,
        roof_tiles_color = (SeamlessTextureWithColor, roofTilesColors),
        #metal = SeamlessTexture,
        metal_color = (SeamlessTextureWithColor, metalColors),
        metal_without_uv = (MaterialWithColor, metalColors),
        #metal_scaled = SeamlessTextureScaled,
        metal_scaled_color = (SeamlessTextureScaledWithColor, metalColors)
    )


def bldgPreRender(building, app):
    element = building.element
    tags = element.tags
    
    # material for walls
    material = building.wallsMaterial
    if not material in ("plaster", "brick", "glass", "mirror"):
        material = "plaster"
    # tb stands for "OSM tag building"
    tb = building.getOsmTagValue("building")
    
    if tb in ("cathedral", "wall") or\
        building.getOsmTagValue("amenity") == "place_of_worship" or\
        building.getOsmTagValue("man_made") or\
        building.getOsmTagValue("barrier"):
        building.setMaterialWalls("%s_color" % material)
    else:
        if material == "glass" or material == "mirror":
            building.setMaterialWalls("glass")
        elif tb == "commercial":
            building.setMaterialWalls("office")
        else:
            building.setMaterialWalls("neoclassical")
    
    # material for roof
    material = building.roofMaterial  
    roofShape = tags.get("roof:shape") or app.defaultRoofShape
    
    if material == "concrete":
        building.setMaterialRoof("concrete_color")
    elif material == "roof_tiles":
        building.setMaterialRoof("roof_tiles_color")
    elif material == "gravel":
        building.setMaterialRoof("gravel_color")
    elif roofShape == "flat":
        building.setMaterialRoof("concrete_color")
    elif roofShape == "gabled":
        building.setMaterialRoof("roof_tiles_color")
    else: # roof:material is metal or metal␣sheet
        if roofShape == "onion":
            building.setMaterialRoof("metal_without_uv")
        elif roofShape == "dome":
            building.setMaterialRoof("metal_scaled_color")
        else:
            building.setMaterialRoof("metal_color")