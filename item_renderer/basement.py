from .container import Container


class Basement(Container):
    
    def render(self, item, indices, uvs):
        self.r.createFace(item.building, indices, uvs)