from defs.building import BldgPolygonFeature


class PolygonSimplification:
    
    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            if polygon.smallFeature:
                currentVector = startVector = polygon.smallFeature.startVector
                while True:
                    feature = currentVector.feature
                    if feature and feature.featureId != BldgPolygonFeature.curved:
                        feature.skipVectors(manager)
                        currentVector = feature.endVector.next
                    else:
                        currentVector = currentVector.next
                    if currentVector is startVector:
                        break