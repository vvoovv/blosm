from .container import Container


class Level(Container):
    
    def __init__(self):
        # The following Python dictionary is used to calculated the number of windows and balconies
        # in the Level pattern
        self.facadePatternInfo = dict(Window=0, Balcony=0)
    
    def init(self, itemRenderers, globalRenderer):
        super().init(itemRenderers, globalRenderer)
    
    def render(self, building, item, indices, uvs):
        self.r.createFace(item.building, indices, uvs)
        if item.markup:
            facadePatternInfo = self.facadePatternInfo
            if not item.materialId:
                # reset <facadePatternInfo>
                for key in facadePatternInfo:
                    facadePatternInfo[key] = 0
                for _item in item.markup:
                    className = _item.__class__.__name__
                    if className in facadePatternInfo:
                        facadePatternInfo[className] += 1
            # get a texture that fits to the Level markup pattern
            textureInfo = self.r.textureStore.getTextureInfo(building, facadePatternInfo)