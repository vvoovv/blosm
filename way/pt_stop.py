from way.item.pt_stop import PtStop


class PtStopManager:
    
    def __init__(self):
        self.layerClass = None
        self.ptStops = []
    
    def parseNode(self, element, elementId):
        self.ptStops.append( PtStop(element) )
    
    def parseWay(self, element, elementId):
        self.ptStops.append( PtStop(element) )
    
    def parseRelation(self, element, elementId):
        return