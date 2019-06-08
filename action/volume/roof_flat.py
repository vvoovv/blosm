from .roof import Roof


class RoofFlat(Roof):
    
    # default roof height
    height = 1
    
    def do(self, item, style, building):
        self.init(item, style)
    
    def calculateRoofLevels(self, footprint, style):
        # the flat roof does not have 
        return