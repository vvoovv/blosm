

class Div:
    """
    A mixin class for Div texture based item renderers
    """
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
        self.bottomRenderer = itemRenderers["Bottom"]
    
    def render(self, item, levelGroup, indices, uvs):
        # If <levelGroup> is given, that actually means that the div item is contained
        # inside a level item. In this case the call to <self.renderLevelGroup(..)>
        # will be made
        if item.markup:
            item.indices = indices
            item.uvs = uvs
            self.renderMarkup(item)
        elif levelGroup:
            self.renderLevelGroup(item, levelGroup, indices, uvs)
        else:
            self.r.createFace(item.building, indices)