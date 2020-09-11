from .. import ItemRenderer


class RoofHipped(ItemRenderer):
    
    def render(self, roofItem):
        
        if sharpSideEdges:
            self.renderSharpSideEdges(roofItem)
        else:
            self.renderIgnoreEdges(roofItem, smoothFaces)