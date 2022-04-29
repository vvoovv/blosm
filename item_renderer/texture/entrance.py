

class Entrance:
    """
    The Entrance renderer is the special case of the <item_renderer.level.Level> when
    an entrance in the only element in the level markup
    
    A mixin class for Entrance texture based item renderers
    """
    
    def render(self, item, indices, uvs):
        face = self.r.createFace(item.building, indices)
        if item.materialId is None:
            self.setMaterialId(
                item,
                # building part
                "entrance",
                uvs
            )
        if item.materialId:
            self.r.setUvs(
                face,
                # overide <uvs>
                ( (0., 0.), (1., 0.), (1., 1.), (0., 1.) ),
                self.r.layer.uvLayerNameFacade
            )
            # claddingTextureInfo = item.materialData[1]
            if item.materialData[1]:
                # set UV-coordinates for the cladding texture
                self.setCladdingUvs(item, face, item.materialData[1], uvs)
                self.setVertexColor(item, face)
        self.r.setMaterial(face, item.materialId)