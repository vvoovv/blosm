import numpy
# matplotlib is used to load an image
import matplotlib.pyplot as plt


class OverlayMixin:

    def finalizeImport(self):
        if self.imageData is None:
            return False
        self.imageData = numpy.reshape(self.imageData, (self.numTilesY*self.tileHeight, self.numTilesX*self.tileWidth, self.numComponents))
        return True
    
    def getTileDataFromImage(self, tilePath):
        return numpy.reshape(plt.imread(tilePath), self.w*self.tileHeight)