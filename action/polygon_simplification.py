from defs.building import BldgPolygonFeature


class PolygonSimplification:
    
    def do(self, manager):

        for building in manager.buildings:
            polygon = building.polygon
            if polygon.smallFeature:
                currentVector = startVector = polygon.smallFeature.startVector
                while True:
                    if currentVector.feature and currentVector.feature.featureId != BldgPolygonFeature.curved:
                        currentVector.feature.skipVectors(manager)
                    currentVector = currentVector.next
                    if currentVector is startVector:
                        break