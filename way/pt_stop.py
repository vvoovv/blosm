from way.item.pt_stop import PtStop


class PtStopManager:
    
    def __init__(self):
        # no layers for this manager
        self.layerClass = self.nodeLayerClass = None
        self.ptStops = []
    
    def parseNode(self, element, elementId):
        self.ptStops.append( PtStop(element) )
    
    def parseWay(self, element, elementId):
        self.ptStops.append( PtStop(element) )
    
    def parseRelation(self, element, elementId):
        return