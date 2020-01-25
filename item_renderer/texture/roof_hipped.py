import math
from .. import ItemRenderer
from ..util import initUvAlongPolygonEdge

from util import zAxis


class RoofHipped(ItemRenderer):
    
    def render(self, roofItem):
        smoothFaces = roofItem.getStyleBlockAttr("faces") is smoothness.Smooth
        sharpSideEdges = smoothFaces and roofItem.getStyleBlockAttr("sharpEdges") is smoothness.Side
        
        if sharpSideEdges:
            self.renderSharpSideEdges(roofItem)
        else:
            self.renderIgnoreEdges(roofItem, smoothFaces)