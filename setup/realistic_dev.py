from style import StyleStore
from style.default import styles

from parse.osm.relation.building import Building as BuildingRelation

from building.manager import BuildingParts, BuildingRelations

from manager.logging import Logger

from building2.manager import RealisticBuildingManager, RealisticBuildingManagerExport
from building2.renderer import BuildingRendererNew, Building

from item.footprint import Footprint

from item_renderer.texture.roof_generatrix import generatrix_dome, generatrix_onion, Center, MiddleOfTheLongesSide

from setup.premium import setup_forests

# item renderers
from item_renderer.texture.base import\
    Facade as FacadeRenderer,\
    Div as DivRenderer,\
    Level as LevelRenderer,\
    CurtainWall as CurtainWallRenderer,\
    Bottom as BottomRenderer,\
    Door as DoorRenderer,\
    RoofFlat as RoofFlatRenderer,\
    RoofFlatMulti as RoofFlatMultiRenderer,\
    RoofProfile as RoofProfileRenderer,\
    RoofGeneratrix as RoofGeneratrixRenderer,\
    RoofPyramidal as RoofPyramidalRenderer

from item_renderer.texture.export import\
    Facade as FacadeRendererExport,\
    Div as DivRendererExport,\
    Level as LevelRendererExport,\
    CurtainWall as CurtainWallRendererExport,\
    Bottom as BottomRendererExport,\
    Door as DoorRendererExport,\
    RoofFlat as RoofFlatRendererExport,\
    RoofFlatMulti as RoofFlatMultiRendererExport,\
    RoofProfile as RoofProfileRendererExport,\
    RoofGeneratrix as RoofGeneratrixRendererExport,\
    RoofPyramidal as RoofPyramidalRendererExport

from action.terrain import Terrain
from action.offset import Offset
from action.volume import Volume


def setup(app, data):
    doExport = app.enableExperimentalFeatures and app.importForExport
    
    styleStore = StyleStore(app, styles=None)

    # comment the next line if logging isn't needed
    Logger(app, data)
    
    if app.buildings:
        buildingParts = BuildingParts()
        buildingRelations = BuildingRelations()
        buildings = RealisticBuildingManagerExport(data, buildingParts)\
            if doExport else\
            RealisticBuildingManager(data, buildingParts)
        
        # Important: <buildingRelation> beform <building>,
        # since there may be a tag building=* in an OSM relation of the type 'building'
        data.addCondition(
            lambda tags, e: isinstance(e, BuildingRelation),
            None,
            buildingRelations
        )
        data.addCondition(
            lambda tags, e: "building" in tags,
            "buildings",
            buildings
        )
        data.addCondition(
            lambda tags, e: "building:part" in tags,
            None,
            buildingParts
        )
        
        # deal with item renderers
        itemRenderers = dict(
            Facade = FacadeRendererExport() if doExport else FacadeRenderer(),
            Div = DivRendererExport() if doExport else DivRenderer(),
            Level = LevelRendererExport() if doExport else LevelRenderer(),
            CurtainWall = CurtainWallRendererExport() if doExport else CurtainWallRenderer(),
            Bottom = BottomRendererExport() if doExport else BottomRenderer(),
            Door = DoorRendererExport() if doExport else DoorRenderer(),
            RoofFlat = RoofFlatRendererExport() if doExport else RoofFlatRenderer(),
            RoofFlatMulti = RoofFlatMultiRendererExport() if doExport else RoofFlatMultiRenderer(),
            RoofProfile = RoofProfileRendererExport() if doExport else RoofProfileRenderer(),
            RoofDome = (RoofGeneratrixRendererExport if doExport else RoofGeneratrixRenderer)(generatrix_dome(7), basePointPosition = Center),
            RoofHalfDome = (RoofGeneratrixRendererExport if doExport else RoofGeneratrixRenderer)(generatrix_dome(7), basePointPosition = MiddleOfTheLongesSide),
            RoofOnion = (RoofGeneratrixRendererExport if doExport else RoofGeneratrixRenderer)(generatrix_onion, basePointPosition = Center),
            RoofPyramidal = RoofPyramidalRendererExport() if doExport else RoofPyramidalRenderer()
        )
        
        br = BuildingRendererNew(app, styleStore, itemRenderers, getStyle=getStyle)
        
        Building.actions = [ Terrain(app, data, br.itemStore, br.itemFactory) ]
        if not app.singleObject:
            Building.actions.append( Offset(app, data, br.itemStore, br.itemFactory) )
        
        volumeAction = Volume(app, data, br.itemStore, br.itemFactory, itemRenderers)
        Footprint.actions = (volumeAction,)
        # <br> stands for "building renderer"
        buildings.setRenderer(br)
        app.managers.append(buildings)
    
    if app.forests:
        setup_forests(app, data)


def getStyle(building, app):
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