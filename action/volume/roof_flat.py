from .roof import Roof


class RoofFlat(Roof):
    
    def do(self, item, style, building):
        self.init(item, style)