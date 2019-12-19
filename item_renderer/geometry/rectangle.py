

class Rectangle:
    
    def __init__(self):
        pass
    
    def getUvs(self, item, numLevelsInFace, numTilesU, numTilesV):
        u = len(item.markup)/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            (u, v),
            (0, v)
        )
    
    @staticmethod
    def getCladdingUvsExport(uvs, textureWidthM, textureHeightM):
        return (
            (uvs[0][0]/textureWidthM, uvs[0][1]/textureHeightM),
            (uvs[1][0]/textureWidthM, uvs[1][1]/textureHeightM),
            (uvs[2][0]/textureWidthM, uvs[2][1]/textureHeightM),
            (uvs[3][0]/textureWidthM, uvs[3][1]/textureHeightM)
        )