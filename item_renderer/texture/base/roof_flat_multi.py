from .item_renderer import ItemRendererMixin
from ..roof_flat_multi import RoofFlatMulti as RoofFlatMultiBase


class RoofFlatMulti(RoofFlatMultiBase, ItemRendererMixin):
    
    def setVertexColor(self, item, faces):
        color = item.getStyleBlockAttrDeep("claddingColor")
        if color:
            for face in faces:
                self.r.setVertexColor(face, color, self.r.layer.vertexColorLayerNameCladding)