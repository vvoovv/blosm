from ...util.geometry_nodes import setGnInput
from . import ItemRenderer
from ..way_properties import wayCategoryProps


class Street(ItemRenderer):
    
    def init(self, globalRenderer):
        super().init(globalRenderer)
        
        self.intersectionRenderer = globalRenderer.itemRenderers["Intersection"]
    
    def requestNodeGroups(self, nodeGroupNames):
        return

    def setNodeGroups(self, nodeGroups):
        return
    
    def setPolyline1ParamsForCorner(self, modifier, connector, setRadius):
        setGnInput(modifier, "Socket_3", connector.item.obj)
        setGnInput(modifier, "Socket_4", connector.leaving)
        if setRadius:
            setGnInput(modifier, "Socket_8", wayCategoryProps[ (connector.item.head if connector.leaving else connector.item.tail).tags["highway"] ]["radius"])

    def setPolyline2ParamsForCorner(self, modifier, connector, setRadius):
        setGnInput(modifier, "Socket_6", connector.item.obj)
        setGnInput(modifier, "Socket_7", connector.leaving)
        if setRadius:
            setGnInput(modifier, "Socket_8", wayCategoryProps[ (connector.item.head if connector.leaving else connector.item.tail).tags["highway"] ]["radius"])
    
    def renderNeighborIntersection(self, intersection, connector, index, modifier):
        street = connector.item
        order = intersection.order
            
        setGnInput(modifier, self.intersectionRenderer.inputCenterlines[order][index][0], street.obj)
        setGnInput(modifier, self.intersectionRenderer.inputWidths[order][index][0], street.head.width if connector.leaving else street.tail.width)
        setGnInput(modifier, self.intersectionRenderer.inputLocations[order][index][0], connector.leaving)