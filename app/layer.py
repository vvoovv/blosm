from mathutils import Vector
from renderer import Renderer
from util.blender import createEmptyObject

class Layer:
    
    def __init__(self, layerId, app):
        self.app = app
        self.id = layerId
        hasTerrain = bool(app.terrain)
        self.singleObject = app.singleObject or hasTerrain
        self.layered = app.layered or hasTerrain
        # instance of BMesh
        self.bm = None
        # Blender object
        self.obj = None
        self.materialIndices = []
        # Blender parent object
        self.parent = None
        self.swModifier = hasTerrain
        # set layer offsets <self.location>, <self.meshZ> and <self.parentLocation>
        # <self.location> is used for a Blender object
        # <self.meshZ> is used for vertices of a BMesh
        # <self.parentLocation> is used for an EMPTY Blender object serving
        # as a parent for Blender objects of the layer
        self.parentLocation = None
        _z = app.layerOffsets[layerId]
        if self.singleObject and self.layered:
            location = Vector((0., 0., _z))
            meshZ = 0.
        elif not self.singleObject and self.layered:
            location = None
            meshZ = 0.
            # it's the only case when <self.parentLocation>
            self.parentLocation = Vector((0., 0., app.layerOffsets[layerId]))
        elif self.singleObject and not self.layered:
            location = None
            meshZ = _z
        elif not self.singleObject and not self.layered:
            location = Vector((0., 0., _z))
            meshZ = 0.
        self.location = location
        self.meshZ = meshZ
        
    def getParent(self):
        # The method is called currently in the single place of the code:
        # in <Renderer.prerender(..)> if (not layer.singleObject and app.layered)
        parent = self.parent
        if not self.parent:
            parent = createEmptyObject(
                self.name,
                self.parentLocation.copy(),
                empty_draw_size=0.01
            )
            parent.parent = Renderer.parent
            self.parent = parent
        return parent
    
    @property
    def name(self):
        return "%s_%s" % (Renderer.name, self.id)