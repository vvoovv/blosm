import os
import numpy
# matplotlib is used to load an image
import PIL.Image


class OverlayMixin:

    def finalizeImport(self):
        if self.imageData is None:
            return False
        self.imageData = numpy.reshape(self.imageData, (self.numTilesY*self.tileHeight, self.numTilesX*self.tileWidth, self.numComponents))
        return True
    
    def getTileDataFromImage(self, tilePath):
        image = PIL.Image.open(tilePath)
        imageFormat = image.format
        tileData = numpy.array(image.getdata())
        image.close()
        
        if self.checkImageFormat:
            # <self.imageExtension> is equal to <png> by defaul
            if imageFormat == "JPEG":
                self.imageExtension = "jpg"
                self.numComponents = 3
                self.initImageData()
                os.rename(
                    tilePath,
                    "%s.jpg" % os.path.splitext(tilePath)[0]
                )
            self.checkImageFormat = False
        
        return numpy.reshape(
            tileData, self.w*self.tileHeight
        )