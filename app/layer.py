from renderer import Renderer
from util import zeroVector
from util.blender import createEmptyObject

class Layer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
        terrain = bool(app.terrain)
        self.singleObject = app.singleObject or terrain
        self.layered = (app.layered if app.singleObject else True) if terrain else app.layered
        # instance of BMesh
        self.bm = None
        # Blender object
        self.obj = None
        self.materialIndices = []
        # Blender parent object
        self.parent = None
        
    def getParent(self):
        parent = self.parent
        if not self.parent:
            parent = createEmptyObject(
                self.name,
                zeroVector(),
                empty_draw_size=0.01
            )
            parent.parent = Renderer.parent
            self.parent = parent
        return parent
    
    @property
    def name(self):
        return "%s_%s" % (Renderer.name, self.id)