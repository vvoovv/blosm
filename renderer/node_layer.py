from . import Renderer
from util.blender import createEmptyObject
from util import zeroVector


class NodeLayer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
        self.bm = None
        self.obj = None
        self.parent = None
        self.parentLocation = zeroVector()

    def getParent(self):
        # The method is called currently in the single place of the code:
        # in <Renderer.preRender(..)> if (not layer.singleObject)
        parent = self.parent
        if not self.parent:
            parent = createEmptyObject(
                self.name,
                self.parentLocation.copy(),
                empty_draw_size = 0.01
            )
            parent.parent = Renderer.parent
            self.parent = parent
        return parent
    
    @property
    def name(self):
        return "%s_%s" % (Renderer.name, self.id)