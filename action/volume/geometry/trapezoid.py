



class Trapezoid:
    
    def getUvs(self, width, heightLeft, heightRight):
        """
        Get flat vertices coordinates on the facade surface (i.e. on the rectangle)
        """
        return ( (0., 0.), (width, 0.), (width, heightRight), (0., heightLeft) )
    
    def getWidth(self, uvs):
        """
        Get width of the facade based on <uvs> calculated in <self.getUvs(..)>
        """
        return uvs[1][0]
    
    def getFinalUvs(self, item, numLevelsInFace, numTilesU, numTilesV):
        u = len(item.markup)/numTilesU
        v = numLevelsInFace/numTilesV
        return (
            (0., 0.),
            (u, 0.),
            (u, v),
            (0, v)
        )