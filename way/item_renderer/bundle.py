from . import ItemRenderer


class Bundle(ItemRenderer):
    
    def init(self, globalRenderer):
        super().init(globalRenderer)
        
        self.intersectionRenderer = globalRenderer.itemRenderers["Intersection"]

    def requestNodeGroups(self, nodeGroupNames):
        return
    
    def setNodeGroups(self, nodeGroups):
        return
    
    def renderNeighborIntersection(self, intersection, connector, index, modifier):
        bundle = connector.item
        order = intersection.order
        
        streetL, streetR = (bundle.streetsHead[-1], bundle.streetsHead[0]) \
            if connector.leaving else \
            (bundle.streetsTail[0], bundle.streetsTail[-1])
        
        streetL_leaving = streetL.pred and streetL.pred.item is bundle
        streetR_leaving = streetR.pred and streetR.pred.item is bundle
            
        modifier[ self.intersectionRenderer.inputCenterlines[order][index][0] ] = streetL.obj
        modifier[ self.intersectionRenderer.inputCenterlines[order][index][1] ] = streetR.obj
        
        modifier[ self.intersectionRenderer.inputWidths[order][index][0] ] = streetL.head.width if streetL_leaving else streetL.tail.width
        modifier[ self.intersectionRenderer.inputWidths[order][index][1] ] = streetR.head.width if streetR_leaving else streetR.tail.width
        
        modifier[ self.intersectionRenderer.inputLocations[order][index][0] ] = streetL_leaving
        modifier[ self.intersectionRenderer.inputLocations[order][index][1] ] = streetR_leaving