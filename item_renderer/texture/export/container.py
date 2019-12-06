from ..container import Container as ContainerBase


class Container(ContainerBase):
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        self.exportMaterials = True

    def renderCladding(self, building, parentItem, face, uvs):
        self.setCladdingUvs(face, uvs)
        super().renderCladding(building, parentItem, face)