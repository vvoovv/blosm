"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
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

from parse.relation.building import Building

from manager import Linestring, Polygon, PolygonAcceptBroken
from renderer import Renderer2d

from building.manager import BuildingManager, BuildingParts, BuildingRelations
from building.renderer import BuildingRenderer

from manager.logging import Logger

from realistic.manager import AreaManager, BaseManager
from realistic.renderer import AreaRenderer, ForestRenderer, WaterRenderer, BareRockRenderer
from realistic.building.renderer import RealisticBuildingRenderer
from realistic.material.renderer import\
    MaterialRenderer, SeamlessTexture, SeamlessTextureWithColor, MaterialWithColor,\
    SeamlessTextureScaledWithColor, SeamlessTextureScaled


def building(tags, e):
    return "building" in tags

def buildingPart(tags, e):
    return "building:part" in tags

def buildingRelation(tags, e):
    return isinstance(e, Building)

def highway(tags, e):
    return "highway" in tags

def railway(tags, e):
    return "railway" in tags

def water(tags, e):
    return tags.get("natural") == "water" or\
        tags.get("waterway") == "riverbank" or\
        tags.get("landuse") == "reservoir"

def coastline(tags, e):
    return tags.get("natural") == "coastline" 

def forest(tags, e):
    return tags.get("natural") == "wood" or tags.get("landuse") == "forest"

#
# "grass", "meadow", "grassland", "farmland"
#
def grass(tags, e):
    return tags.get("landuse") == "grass"
def meadow(tags, e):
    return tags.get("landuse") == "meadow"
def grassland(tags, e):
    return tags.get("natural") == "grassland"
def farmland(tags, e):
    return tags.get("landuse") == "farmland"

#
# "scrub", "heath"
#
def scrub(tags, e):
    return tags.get("natural") == "scrub"
def heath(tags, e):
    return tags.get("natural") == "heath"

#
# "marsh", "reedbed", "bog", "swamp"
#
def marsh(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "marsh"
def reedbed(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "reedbed"
def bog(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "bog"
def swamp(tags, e):
    return tags.get("natural") == "wetland" and tags.get("wetland") == "swamp"

#
# "glacier"
#
def glacier(tags, e):
    return tags.get("natural") == "glacier"

#
# "bare_rock:
#
def bare_rock(tags, e):
    return tags.get("natural") == "bare_rock"

#
# "scree", "shingle"
# "beach" with "gravel" or "pebbles"
#
def scree(tags, e):
    return tags.get("natural") == "scree"
def shingle(tags, e):
    natural = tags.get("natural")
    return natural == "shingle" or (natural == "beach" and tags.get("surface") in ("gravel", "pebbles"))

#
# "sand"
# "beach" with "sand"
#
def sand(tags, e):
    natural = tags.get("natural")
    # The condition is added after shingle(..),
    # so any value of <surface> for <natural=beach> or its absence is
    # considered as <surface=sand>
    return natural == "sand" or natural == "beach"


def setup(app, osm):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    areaRenderers = {}
    
    # create managers
    linestring = Linestring(osm)
    polygon = Polygon(osm)
    polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    if app.buildings:
        if app.mode is app.twoD:
            osm.addCondition(building, "buildings", polygon)
        else: # 3D
            buildingParts = BuildingParts()
            buildingRelations = BuildingRelations()
            buildings = BuildingManager(osm, buildingParts)
            
            # Important: <buildingRelation> beform <building>,
            # since there may be a tag building=* in an OSM relation of the type 'building'
            osm.addCondition(
                buildingRelation, None, buildingRelations
            )
            osm.addCondition(
                building, None, buildings
            )
            osm.addCondition(
                buildingPart, None, buildingParts
            )
            # set building renderer
            if app.mode is app.realistic:
                br = RealisticBuildingRenderer(
                    app,
                    "buildings",
                    bldgPreRender = bldgPreRender,
                    materials = getMaterials()
                )
            else:
                br = BuildingRenderer(app, "buildings")
            # <br> stands for "building renderer"
            buildings.setRenderer(br)
            app.managers.append(buildings)
    
    if app.highways:
        osm.addCondition(highway, "highways", linestring)
    if app.railways:
        osm.addCondition(railway, "railways", linestring)
    if app.water:
        osm.addCondition(water, "water", polygonAcceptBroken)
        osm.addCondition(coastline, "coastlines", linestring)
        areaRenderers["water"] = WaterRenderer()
        areaRenderers["coastlines"] = None
    if app.forests:
        osm.addCondition(forest, "forest", polygon)
        areaRenderers["forest"] = ForestRenderer()
    if app.vegetation:
        # "grass", "meadow", "grassland", "farmland"
        osm.addCondition(grass, "grass", polygon)
        osm.addCondition(meadow, "meadow", polygon)
        osm.addCondition(grassland, "grassland", polygon)
        osm.addCondition(farmland, "farmland", polygon)
        # "scrub", "heath"
        osm.addCondition(scrub, "scrub", polygon)
        osm.addCondition(heath, "heath", polygon)
        # "marsh", "reedbed", "bog", "swamp"
        osm.addCondition(marsh, "marsh", polygon)
        osm.addCondition(reedbed, "reedbed", polygon)
        osm.addCondition(bog, "bog", polygon)
        osm.addCondition(swamp, "swamp", polygon)
    if False:
    #if app.otherAreas:
        osm.addCondition(glacier, "glacier", polygon)
    if False:
        osm.addCondition(bare_rock, "bare_rock", polygon)
        areaRenderers["bare_rock"] = BareRockRenderer()
    
    if False:
        osm.addCondition(scree, "scree", polygon)
        osm.addCondition(shingle, "shingle", polygon)
        osm.addCondition(sand, "sand", polygon)
        
    
    numConditions = len(osm.conditions)
    if app.buildings:
        # 3D buildings aren't processed by AreaManager or BaseManager
        numConditions -= 3
    if numConditions:
        if app.terrainObject:
            m = AreaManager(osm, app, AreaRenderer(), **areaRenderers)
            m.setRenderer(Renderer2d(app, applyMaterial=False))
        else:
            m = BaseManager(osm)
            m.setRenderer(Renderer2d(app))
        app.managers.append(m)


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


class FacadeSeamlessTexture(MaterialRenderer):
    
    uvLayer = "data.1"
    
    def __init__(self, renderer, baseMaterialName):
        super().__init__(renderer, baseMaterialName)
        self.materialName2 = "%s_with_ground_level" % baseMaterialName
        
    def init(self):
        self.ensureUvLayer(self.uvLayer)
        self.setupMaterials(self.materialName)
        self.setupMaterials(self.materialName2)
        
    def renderWalls(self, face):
        # building
        b = self.b
        if b.z1:
            self.setData(face, self.uvLayer, b.numLevels)
            self.setMaterial(face, self.materialName)
        else:
            self.setData(face, self.uvLayer, b.levelHeights)
            self.setMaterial(face, self.materialName2)