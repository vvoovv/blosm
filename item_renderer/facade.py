from .container import Container


class Facade(Container):
    
    def __init__(self):
        super().__init__()
    
    def render(self, footprint, building):
        polygon = footprint.polygon
        