from style import StyleStore

from parse.osm.relation.building import Building as BuildingRelation

from building.manager import BuildingParts, BuildingRelations

from manager.logging import Logger

from building2.manager import RealisticBuildingManager
from building2.renderer import BuildingRendererNew, Building
from building2.layer import RealisticBuildingLayer

from item.footprint import Footprint

# item renderers
from item_renderer.test import\
    GeometryRenderer, GeometryRendererRoofWithSides, GeometryRendererFacade, GeometryRendererRoofFlat

from action.terrain import Terrain
from action.offset import Offset
from action.volume import Volume


import bpy


def redefineMethods():
    #
    # redefine BuildingManager.render(..) to render only a part of buildings
    #
    from building.manager import BuildingManager
    def bmRender(self):
        numBuildings = len(self.buildings)
        for i in range(numBuildings):
        #for i in range(0, numBuildings):
            building = self.buildings[i]
            print("%s:%s" % (i, building.outline.tags["id"]))
            self.renderer.render(self.buildings[i], self.osm)
    BuildingManager.render = bmRender
    
    
    #
    # augment BuildingRendererNew.render(..)
    #
    import math
    from parse.osm.way import Way
    
    _brRender = BuildingRendererNew.render
    def brRender(self, buildingP, data):
        projection = data.projection
        element = buildingP.outline
        if isinstance(element, Way):
            node = data.nodes[element.nodes[0]]
        else:
            # the case of OSM relation
            node = data.nodes[
                next( (element.ls[0] if isinstance(element.ls, list) else element.ls).nodeIds(data) )
            ]
        
        projection.lat = node.lat
        projection.lon = node.lon
        projection.latInRadians = math.radians(projection.lat)
        _brRender(self, buildingP, data)
    
    BuildingRendererNew.render = brRender
    
    
    #
    # augment BlenderApp.clean(..)
    #
    from app.blender import BlenderApp
    
    _clean = BlenderApp.clean
    def clean(self):
        self.log.close()
        _clean(self)
    
    BlenderApp.clean = clean
    
    
    #
    # redefine Node.getData(..) (never cache the projected coordinates)
    #
    from parse.osm.node import Node
    
    def getData(self, osm):
        """
        Get projected coordinates
        """
        return osm.projection.fromGeographic(self.lat, self.lon)
    Node.getData = getData
    
    
    #
    # redefine RoofHipped.render(..) to catch exceptions inside lib.bpypolyskel.polygonize(..)
    #
    from action.volume.roof_hipped import RoofHipped
    
    _rhGenerateRoof = RoofHipped.generateRoof
    def rhGenerateRoof(self, footprint, roofItem, firstVertIndex):
        roofItem.exception = None
        try:
            _rhGenerateRoof(self, footprint, roofItem, firstVertIndex)
        except Exception as e:
            roofItem.exception = "%s-%s" % (e.__class__.__name__, str(e))
    RoofHipped.generateRoof = rhGenerateRoof
    
    
    #
    # redefine RoofHipped.generateRoofQuadrangle(..) to set <roofItem.exception>
    #
    _rhGenerateRoofQuadrangle = RoofHipped.generateRoofQuadrangle
    def rhGenerateRoofQuadrangle(self, footprint, roofItem, firstVertIndex):
        roofItem.exception = None
        _rhGenerateRoofQuadrangle(self, footprint, roofItem, firstVertIndex)
    RoofHipped.generateRoofQuadrangle = rhGenerateRoofQuadrangle
    
    
    #
    # redefine RoofHipped.render(..) to catch exceptions inside lib.bpypolyskel.polygonize(..)
    #
    from action.volume.roof_hipped_multi import RoofHippedMulti
    
    _rhmGenerateRoof = RoofHippedMulti.generateRoof
    def rhmGenerateRoof(self, footprint, roofItem, firstVertIndex):
        roofItem.exception = None
        try:
            _rhmGenerateRoof(self, footprint, roofItem, firstVertIndex)
        except Exception as e:
            roofItem.exception = "%s-%s" % (e.__class__.__name__, str(e))
    RoofHippedMulti.generateRoof = rhmGenerateRoof


def setup(app, data):
    if not hasattr(app, "methodsRedefined"):
        redefineMethods()
        app.methodsRedefined = True
    
    # prevent extent calculation
    bpy.context.scene["lat"] = 0.
    bpy.context.scene["lon"] = 0.
    # create a log
    app.log = open("D://tmp/log.txt", 'w')
    
    styleStore = StyleStore(app, styles=None)

    # comment the next line if logging isn't needed
    Logger(app, data)
    
    if app.buildings:
        buildingParts = BuildingParts()
        buildingRelations = BuildingRelations()
        buildings = RealisticBuildingManager(data, app, buildingParts, RealisticBuildingLayer)
        
        # Important: <buildingRelation> beform <building>,
        # since there may be a tag building=* in an OSM relation of the type 'building'
        data.addCondition(
            lambda tags, e: isinstance(e, BuildingRelation),
            None,
            buildingRelations
        )
        data.addCondition(
            lambda tags, e: "building" in tags or "building:part" in tags,
            "buildings",
            buildings
        )
        
        # deal with item renderers
        itemRenderers = dict(
            Facade = GeometryRendererFacade(),
            Div = GeometryRenderer(),
            Level = GeometryRenderer(),
            CurtainWall = GeometryRenderer(),
            Bottom = GeometryRenderer(),
            Door = GeometryRenderer(),
            RoofFlat = GeometryRendererRoofFlat(),
            RoofFlatMulti = GeometryRenderer(),
            RoofProfile = GeometryRenderer(),
            RoofDome = GeometryRenderer(),
            RoofHalfDome = GeometryRenderer(),
            RoofOnion = GeometryRenderer(),
            RoofPyramidal = GeometryRenderer(),
            RoofHipped = GeometryRendererRoofWithSides()
        )
        
        br = BuildingRendererNew(app, styleStore, itemRenderers, getStyle=getStyle)
        
        Building.actions = []
        # <app.terrain> isn't yet set at this pooint, so we use the string <app.terrainObject> instead
        if app.terrainObject:
            Building.actions.append( Terrain(app, data, br.itemStore, br.itemFactory) )
        if not app.singleObject:
            Building.actions.append( Offset(app, data, br.itemStore, br.itemFactory) )
        
        volumeAction = Volume(app, data, br.itemStore, br.itemFactory, itemRenderers)
        Footprint.actions = (volumeAction,)
        # <br> stands for "building renderer"
        buildings.setRenderer(br)


def getStyle(building, app):
    #if building["id"] in removeFromTest:
    #    return
    #return "mid rise apartments zaandam"
    #return "high rise mirrored glass"
    buildingTag = building["building"]
    
    if buildingTag in ("commercial", "office"):
        return "high rise"
    
    if buildingTag in ("house", "detached"):
        return "single family house"
    
    if buildingTag in ("residential", "apartments", "house", "detached"):
        return "residential"
    
    if building["amenity"] == "place_of_worship":
        return "place of worship"
    
    if building["man_made"] or building["barrier"] or buildingTag=="wall":
        return "man made"
    
    buildingArea = building.area()
    
    if buildingArea < 20.:
        return "small structure"
    elif buildingArea < 200.:
        return "single family house"
    
    return "high rise"