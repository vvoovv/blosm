from .roof_hipped import RoofHipped


class RoofHippedMulti(RoofHipped):
    
    def __init__(self):
        super().__init__()
        self.innerPolygons = []
    
    def init(self):
        super().init()
        self.innerPolygons.clear()