from manager import Manager
from .item_renderer import ItemRendererMixin
from ..roof_flat_multi import RoofFlatMulti as RoofFlatMultiBase


class RoofFlatMulti(RoofFlatMultiBase, ItemRendererMixin):
    
    def setVertexColor(self, item, faces):
        color = Manager.normalizeColor(item.getStyleBlockAttrDeep("claddingColor"))
        if color:
            color = Manager.getColor(color)
            for face in faces:
                self.r.setVertexColor(face, color, self.r.layer.vertexColorLayerNameCladding)