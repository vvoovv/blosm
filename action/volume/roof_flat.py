from .roof import Roof


class RoofFlat(Roof):
    
    # default roof height
    height = 1.
    
    def __init__(self, data, itemStore, itemFactory):
        super().__init__(data, itemStore, itemFactory)
        self.hasRoofLevels = False
    
    def do(self, item, style, building):
        self.init(item, style)
    
    def calculateRoofLevels(self, footprint, style):
        # the flat roof does not have 
        return