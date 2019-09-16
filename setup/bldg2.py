from style import StyleStore
from style.default import styles

from parse.osm.relation.building import Building as BuildingRelation

from building.manager import BuildingParts, BuildingRelations

from manager.logging import Logger

from realistic.building.manager import RealisticBuildingManager
from building2.renderer import BuildingRendererNew, Building

from item.footprint import Footprint

# item renderers
from item_renderer.facade import Facade as FacadeRenderer
from item_renderer.div import Div as DivRenderer
from item_renderer.level import Level as LevelRenderer
from item_renderer.basement import Basement as BasementRenderer

from action.terrain import Terrain
from action.volume import Volume


def setup(app, data):
    styleStore = StyleStore()
    for styleName in styles:
        styleStore.add(styleName, styles[styleName])

    # comment the next line if logging isn't needed
    Logger(app, data)
    
    if app.buildings:
        buildingParts = BuildingParts()
        buildingRelations = BuildingRelations()
        buildings = RealisticBuildingManager(data, buildingParts)
        
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
        # set building renderer
        #br = RealisticBuildingRenderer(
        #    app,
        #    bldgPreRender = bldgPreRender,
        #    materials = getMaterials()
        #)
        
        br = BuildingRendererNew(app, styleStore, getStyle=getStyle)
        
        # deal with item renderers
        itemRenderers = dict(
            Facade = FacadeRenderer(),
            Div = DivRenderer(),
            Level = LevelRenderer(),
            Basement = BasementRenderer()
        )
        for item in itemRenderers:
            itemRenderers[item].init(itemRenderers, br)
        
        Building.actions = (Terrain(app, data, br.itemStore, br.itemFactory),)
        
        volumeAction = Volume(app, data, br.itemStore, br.itemFactory)
        volumeAction.setRenderer(itemRenderers["Facade"])
        Footprint.actions = (volumeAction,)
        # <br> stands for "building renderer"
        buildings.setRenderer(br)
        app.managers.append(buildings)
    
    #if app.forests:
    #    setup_forests(app, osm)


def getStyle(building, app):
    return "mid rise residential zaandam"