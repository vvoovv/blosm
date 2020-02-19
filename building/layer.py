from renderer.layer import MeshLayer
from util import zeroVector


class BuildingLayer(MeshLayer):
    
    def __init__(self, layerId, app):
        super().__init__(layerId, app)
        # does the layer represents an area (natural or landuse)?
        self.area = False
    
    def init(self):
        super().init()
        
        # set layer offsets <layer.location>, <layer.meshZ> and <layer.parentLocation> to zero
        self.location = None
        self.meshZ = 0.
        if self.parentLocation:
            self.parentLocation[2] = 0.
        else:
            self.parentLocation = zeroVector()
        
        if self.app.terrain:
            # the attribute <singleObject> of the buildings layer doesn't depend on availability of a terrain
            # no need to apply any Blender modifier for buildings
            self.modifiers = False
            # no need to slice Blender mesh
            self.sliceMesh = False