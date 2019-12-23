from .container import Container
from ..door import Door as DoorBase


class Door(DoorBase, Container):
        
    def __init__(self):
        # a reference to the Container class used in the parent classes
        self.Container = Container
        Container.__init__(self, exportMaterials=True)
        DoorBase.__init__(self)

    def init(self, itemRenderers, globalRenderer):
        Container.init(self, itemRenderers, globalRenderer)
        self.exporter = globalRenderer.materialExportManager.doorExporter

    def getFacadeMaterialId(self, item, facadeTextureInfo, claddingTextureInfo):
        color = self.getCladdingColor(item)
        return "door_%s_%s_%s" % (claddingTextureInfo["material"], color, facadeTextureInfo["name"])\
            if claddingTextureInfo and color\
            else facadeTextureInfo["name"]

    def render(self, building, levelGroup, parentItem, indices, uvs):
        face = self.r.createFace(building, indices)
        item = levelGroup.item
        if item.materialId is None:
            self.setMaterialId(
                item,
                building,
                # building part
                "door",
                uvs,
                # item renderer
                self
            )
        if item.materialId:
            self.r.setUvs(
                face,
                # we assume that the face is a rectangle
                (
                    (0., 0.), (1., 0.), (1., 1.), (0., 1.)
                ),
                self.r.layer.uvLayerNameFacade
            )
        self.r.setMaterial(face, item.materialId)