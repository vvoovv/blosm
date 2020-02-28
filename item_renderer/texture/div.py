

class Div:
    """
    A mixin class for Div texture based item renderers
    """
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
        self.bottomRenderer = itemRenderers["Bottom"]
    
    def render(self, item, indices, uvs):
        if item.markup:
            item.indices = indices
            item.uvs = uvs
            self.renderMarkup(item)
        else:
            self.r.createFace(item.building, indices)