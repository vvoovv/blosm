import math
from .. import ItemRenderer
from ..util import initUvAlongPolygonEdge

from util import zAxis


class RoofHipped(ItemRenderer):
    
    def render(self, roofItem):
        
        if sharpSideEdges:
            self.renderSharpSideEdges(roofItem)
        else:
            self.renderIgnoreEdges(roofItem, smoothFaces)