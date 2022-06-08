from building.manager import BuildingParts, BuildingRelations


def conditionsSkipWays(tags, e):
    if tags.get("area") == "yes" or tags.get("tunnel") == "yes" or tags.get("ice_road") == "yes":
        e.valid = False
        return True
    return False


class Setup:
    
    def __init__(self, app, osm):
        self.app = app
        self.osm = osm
        
        self.wayManager = None
        
        self.featureDetectionAction = None
        self.skipFeaturesAction = None
        self.unskipFeaturesAction = None
    
    def detectFeatures(self, simplifyPolygons):
        from action.curved_features import CurvedFeatures
        from action.straight_angles import StraightAngles
        
        buildingManager = self.buildingManager
        buildingManager.addAction(CurvedFeatures())
        buildingManager.addAction(StraightAngles())
        buildingManager.addAction(self.getFeatureDetectionAction(simplifyPolygons))
    
    def facadeVisibility(self, facadeVisibilityAction):
        if not facadeVisibilityAction:
            from action.facade_visibility import FacadeVisibilityBlender
            facadeVisibilityAction = FacadeVisibilityBlender()
        
        self.buildingManager.addAction(facadeVisibilityAction)
    
    def classifyFacades(self, facadeVisibilityAction=None):
        from action.facade_classification import FacadeClassification
        
        self.facadeVisibility(facadeVisibilityAction)
        
        self.buildingManager.addAction(
            FacadeClassification(self.getUnskipFeaturesAction())
        )
    
    def buildings(self):
        self.osm.addCondition(
            lambda tags, e: "building" in tags,
            "buildings",
            self.buildingManager
        )
    
    def skipWays(self, skipFunction=None):
        if not skipFunction:
            skipFunction = conditionsSkipWays
        self.osm.addCondition(skipFunction)
    
    def roadsAndPaths(self):
        osm = self.osm
        wayManager = self.getWayManager()
        
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("motorway", "motorway_link"),
            "roads_motorway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("trunk", "trunk_link"),
            "roads_trunk",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("primary", "primary_link"),
            "roads_primary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("secondary", "secondary_link"),
            "roads_secondary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("tertiary", "tertiary_link"),
            "roads_tertiary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "unclassified",
            "roads_unclassified",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("residential", "living_street"),
            "roads_residential",
            wayManager
        )
        # footway to optimize the walk through conditions
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("footway", "path"),
            "paths_footway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "service",
            "roads_service",
            wayManager
        )
        # filter out pedestrian areas for now
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "pedestrian" and not tags.get("area") and not tags.get("area:highway"),
            "roads_pedestrian",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "track",
            "roads_track",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "steps",
            "paths_steps",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "cycleway",
            "paths_cycleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "bridleway",
            "paths_bridleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("road", "escape", "raceway"),
            "roads_other",
            wayManager
        )
    
    def railways(self):
        self.osm.addCondition(
            lambda tags, e: tags.get("railway") in ("rail", "tram", "subway", "light_rail", "funicular", "monorail"),
            "railways",
            self.getWayManager()
        )
    
    def getFeatureDetectionAction(self, simplifyPolygons):
        if not self.featureDetectionAction:
            from action.feature_detection import FeatureDetection
            self.featureDetectionAction = FeatureDetection(
                self.getSkipFeaturesAction() if simplifyPolygons else None
            )
        return self.featureDetectionAction
    
    def getSkipFeaturesAction(self):
        if not self.skipFeaturesAction:
            from action.skip_features import SkipFeatures
            self.skipFeaturesAction = SkipFeatures()
        return self.skipFeaturesAction
    
    def getUnskipFeaturesAction(self):
        if not self.unskipFeaturesAction:
            from action.unskip_features import UnskipFeatures
            self.unskipFeaturesAction = UnskipFeatures()
        return self.unskipFeaturesAction
    
    def getWayManager(self):
        if not self.wayManager:
            from way.manager import WayManager
            self.wayManager = WayManager(self.osm, self.app)
        return self.wayManager


class SetupBlender(Setup):
    
    def __init__(self, app, osm):
        super().__init__(app, osm)
        
        self.doExport = app.enableExperimentalFeatures and app.importForExport
    
    def buildingsRealistic(self, getStyle):
        from building2.manager import RealisticBuildingManager
        from building2.renderer import BuildingRendererNew
        from style import StyleStore
        
        if self.doExport:
            from building2.layer import RealisticBuildingLayerExport as RealisticBuildingLayer
        else:
            from building2.layer import RealisticBuildingLayerBase as RealisticBuildingLayer
        
        buildingParts = BuildingParts()
        buildingRelations = BuildingRelations()
        self.buildingManager = buildingManager = RealisticBuildingManager(
            self.osm,
            self.app,
            buildingParts,
            RealisticBuildingLayer
        )
        
        self.conditionsBuildings(buildingManager, buildingParts, buildingRelations)
        
        itemRenderers = self.itemRenderers()
        
        buildingRenderer = BuildingRendererNew(
            self.app,
            StyleStore(self.app, styles=None),
            itemRenderers,
            getStyle=getStyle
        )
        
        self.actionsBuildings(buildingManager, buildingRenderer, itemRenderers)
        
        buildingManager.setRenderer(buildingRenderer)
    
    def conditionsBuildings(self, buildingManager, buildingParts, buildingRelations):
        from parse.osm.relation.building import Building as BuildingRelation
        
        osm = self.osm
        # Important: <buildingRelation> beform <building>,
        # since there may be a tag building=* in an OSM relation of the type 'building'
        osm.addCondition(
            lambda tags, e: isinstance(e, BuildingRelation),
            None,
            buildingRelations
        )
        osm.addCondition(
            lambda tags, e: "building" in tags,
            "buildings",
            buildingManager
        )
        osm.addCondition(
            lambda tags, e: "building:part" in tags,
            None,
            buildingParts
        )
    
    def itemRenderers(self):
        from item_renderer.texture.roof_generatrix import generatrix_dome, generatrix_onion, Center, MiddleOfTheLongesSide
        
        if self.doExport:
            from item_renderer.texture.export import\
                Facade as FacadeRendererExport,\
                Div as DivRendererExport,\
                Level as LevelRendererExport,\
                Top as TopRendererExport,\
                Bottom as BottomRendererExport,\
                Entrance as EntranceRendererExport,\
                RoofFlat as RoofFlatRendererExport,\
                RoofFlatMulti as RoofFlatMultiRendererExport,\
                RoofProfile as RoofProfileRendererExport,\
                RoofGeneratrix as RoofGeneratrixRendererExport,\
                RoofPyramidal as RoofPyramidalRendererExport,\
                RoofHipped as RoofHippedRendererExport
            
            itemRenderers = dict(
                Facade = FacadeRendererExport(),
                Div = DivRendererExport(),
                Level = LevelRendererExport(),
                Top = TopRendererExport(),
                Bottom = BottomRendererExport(),
                Entrance = EntranceRendererExport(),
                RoofFlat = RoofFlatRendererExport(),
                RoofFlatMulti = RoofFlatMultiRendererExport(),
                RoofProfile = RoofProfileRendererExport(),
                RoofDome = RoofGeneratrixRendererExport(generatrix_dome(7), basePointPosition = Center),
                RoofHalfDome = RoofGeneratrixRendererExport(generatrix_dome(7), basePointPosition = MiddleOfTheLongesSide),
                RoofOnion = RoofGeneratrixRendererExport(generatrix_onion, basePointPosition = Center),
                RoofPyramidal = RoofPyramidalRendererExport(),
                RoofHipped = RoofHippedRendererExport()
            )
        else:
            from item_renderer.texture.base import\
                Facade as FacadeRenderer,\
                Div as DivRenderer,\
                Level as LevelRenderer,\
                Top as TopRenderer,\
                Bottom as BottomRenderer,\
                Entrance as EntranceRenderer,\
                RoofFlat as RoofFlatRenderer,\
                RoofFlatMulti as RoofFlatMultiRenderer,\
                RoofProfile as RoofProfileRenderer,\
                RoofGeneratrix as RoofGeneratrixRenderer,\
                RoofPyramidal as RoofPyramidalRenderer,\
                RoofHipped as RoofHippedRenderer
            
            itemRenderers = dict(
                Facade = FacadeRenderer(),
                Div = DivRenderer(),
                Level = LevelRenderer(),
                Top = TopRenderer(),
                Bottom = BottomRenderer(),
                Entrance = EntranceRenderer(),
                RoofFlat = RoofFlatRenderer(),
                RoofFlatMulti = RoofFlatMultiRenderer(),
                RoofProfile = RoofProfileRenderer(),
                RoofDome = RoofGeneratrixRenderer(generatrix_dome(7), basePointPosition = Center),
                RoofHalfDome = RoofGeneratrixRenderer(generatrix_dome(7), basePointPosition = MiddleOfTheLongesSide),
                RoofOnion = RoofGeneratrixRenderer(generatrix_onion, basePointPosition = Center),
                RoofPyramidal = RoofPyramidalRenderer(),
                RoofHipped = RoofHippedRenderer()
            )
        
        return itemRenderers
    
    def actionsBuildings(self, buildingManager, buildingRenderer, itemRenderers):
        from action.volume import Volume
        from action.facade_boolean import FacadeBoolean
        from action.facade_classification import FacadeClassificationPart
        
        app = self.app
        osm = self.osm
        
        # <app.terrain> isn't yet set at this pooint, so we use the string <app.terrainObject> instead
        if app.terrainObject:
            from action.terrain import Terrain
            buildingRenderer.buildingActions.append( Terrain(app, osm, buildingRenderer.itemStore) )
        if not app.singleObject:
            from action.offset import Offset
            buildingRenderer.buildingActions.append( Offset(app, osm, buildingRenderer.itemStore) )
        
        volumeAction = Volume(buildingManager, buildingRenderer.itemStore, itemRenderers)
        buildingRenderer.footprintActions.append(volumeAction)
        
        # "rev" stands for "render extruded volumes"
        buildingRenderer.revActions.append(
            FacadeBoolean()
        )
        buildingRenderer.revActions.append(
            FacadeClassificationPart()
        )