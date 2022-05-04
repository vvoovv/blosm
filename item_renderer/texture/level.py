

class Level:
    
    def __init__(self):
        pass
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)


class CurtainWall(Level):
    
    def __init__(self):
        super().__init__()
        
        self.claddingTexture = False