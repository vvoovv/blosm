from ...util.geometry_nodes import setGnInput
from . import ItemRenderer
from ..way_properties import wayCategoryProps


class Bundle(ItemRenderer):
    
    def init(self, globalRenderer):
        super().init(globalRenderer)
        
        self.intersectionRenderer = globalRenderer.itemRenderers["Intersection"]

    def requestNodeGroups(self, nodeGroupNames):
        return
    
    def setNodeGroups(self, nodeGroups):
        return

    def setPolyline1ParamsForCorner(self, modifier, connector, setRadius):
        intersection = connector.intersection
        bundle = connector.item
        # we need the rightmost street in the bundle for the 1st polyline
        street = bundle.streetsHead[0] if connector.leaving else bundle.streetsTail[-1]
        leaving = bool(street.pred and street.pred.intersection is intersection)
        setGnInput(modifier, "Socket_3", street.obj)
        setGnInput(modifier, "Socket_4", leaving)
        if setRadius:
            setGnInput(modifier, "Socket_8", wayCategoryProps[ (street.head if leaving else street.tail).tags["highway"] ]["radius"])

    def setPolyline2ParamsForCorner(self, modifier, connector, setRadius):
        intersection = connector.intersection
        bundle = connector.item
        # we need the leftmost street in the bundle for the 2nd polyline
        street = bundle.streetsHead[-1] if connector.leaving else bundle.streetsTail[0]
        leaving = bool(street.pred and street.pred.intersection is intersection)
        setGnInput(modifier, "Socket_6", street.obj)
        setGnInput(modifier, "Socket_7", leaving)
        if setRadius:
            setGnInput(modifier, "Socket_8", wayCategoryProps[ (street.head if leaving else street.tail).tags["highway"] ]["radius"])
    
    def renderNeighborIntersection(self, intersection, connector, index, modifier):
        bundle = connector.item
        order = intersection.order
        
        streetL, streetR = (bundle.streetsHead[-1], bundle.streetsHead[0]) \
            if connector.leaving else \
            (bundle.streetsTail[0], bundle.streetsTail[-1])
        
        streetL_leaving = bool(streetL.pred and streetL.pred.intersection is intersection)
        streetR_leaving = bool(streetR.pred and streetR.pred.intersection is intersection)
            
        setGnInput(modifier, self.intersectionRenderer.inputCenterlines[order][index][0], streetL.obj)
        setGnInput(modifier, self.intersectionRenderer.inputCenterlines[order][index][1], streetR.obj)
        
        setGnInput(modifier, self.intersectionRenderer.inputWidths[order][index][0], streetL.head.width if streetL_leaving else streetL.tail.width)
        setGnInput(modifier, self.intersectionRenderer.inputWidths[order][index][1], streetR.head.width if streetR_leaving else streetR.tail.width)
        
        setGnInput(modifier, self.intersectionRenderer.inputLocations[order][index][0], streetL_leaving)
        setGnInput(modifier, self.intersectionRenderer.inputLocations[order][index][1], streetR_leaving)