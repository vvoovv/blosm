from .container import Container


class Level(Container):
    
    def render(self, item, indices, uvs):
        self.r.createFace(item.building, indices, uvs)
        return
        if item.markup or item.styleBlock.markup:
            item.indices = indices
            item.uvs = uvs
            self.renderMarkup(item)
        else:
            self.r.createFace(item.building, indices, uvs)