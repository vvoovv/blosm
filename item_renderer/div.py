from .container import Container


class Div(Container):
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
        self.levelRenderer = itemRenderers["Level"]
    
    def render(self, item, indices, uvs):
        if item.markup or item.styleBlock.markup:
            item.indices = indices
            item.uvs = uvs
            self.renderMarkup(item)
        else:
            self.r.createFace(item.building, indices, uvs)