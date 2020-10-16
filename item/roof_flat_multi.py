from .roof_flat import RoofFlat


class RoofFlatMulti(RoofFlat):
    
    def __init__(self):
        super().__init__()
        self.innerPolygons = []
    
    def init(self):
        super().init()
        self.innerPolygons.clear()