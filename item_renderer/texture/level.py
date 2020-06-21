

class Level:
    
    def __init__(self):
        pass
    
    def init(self, itemRenderers, globalRenderer):
        self.Container.init(self, itemRenderers, globalRenderer)
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)


class CurtainWall(Level):
    
    def __init__(self):
        super().__init__()
        
        self.noCladdingTexture = True