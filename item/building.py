

class Building:
    """
    A class representing the building for the renderer
    """
    
    def __init__(self, data):
        self.data = data
        
        self.verts = []
        # counterparts for <self.verts> in the BMesh
        self.bmVerts = []
        # A cache to store different stuff:
        # attributes evaluated per building rather than per footprint, cladding texture info
        self._cache = {}
        
        # offset for vertices
        self.offsetVertex = None
        # Offset for a Blender object. Used only for <app.singleObject=True>
        self.offsetBlenderObject = None
        # Instance of item.footprint.Footprint, it's only used if the building definition
        # in the data model doesn't contain building parts, i.e. the building is defined completely
        # by its outline
        self.footprint = None
        self._area = 0.
        # altitude difference for the building footprint projected on the terrain
        self.altitudeDifference = 0.
    
    def setStyleMeta(self, style):
        if style.meta:
            for attr in style.meta.attrs:
                setattr(self, attr, style.meta.attrs[attr])