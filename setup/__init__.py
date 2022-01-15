

def conditionsSkipWays(tags, e):
    if tags.get("area") == "yes" or tags.get("tunnel") == "yes" or tags.get("ice_road") == "yes":
        e.valid = False
        return True
    return False


class Setup:
    
    def __init__(self, osm):
        self.osm = osm
        
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
    
    def classifyFacades(self, facadeVisibilityAction):
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
        wayManager = self.wayManager
        
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
            lambda tags, e: "railway" in tags,
            "railways",
            self.wayManager
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