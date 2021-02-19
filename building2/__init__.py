from mathutils import Vector
import parse
from util.polygon import Polygon


class Building:
    """
    A class representing the building for the renderer
    """
    
    actions = []
    
    def __init__(self):
        self.verts = []
        # counterparts for <self.verts> in the BMesh
        self.bmVerts = []
        # A cache to store different stuff:
        # attributes evaluated per building rather than per footprint, cladding texture info
        self._cache = {}
        # <self.outlinePolygon> is used only in the case if the buildings has parts
        self.outlinePolygon = Polygon()
    
    def init(self, outline):
        self.verts.clear()
        self.bmVerts.clear()
        # <outline> is an instance of the class as defined by the data model (e.g. parse.osm.way.Way) 
        self.outline = outline
        if self.outlinePolygon.allVerts:
            self.outlinePolygon.clear()
        self.offset = None
        # Instance of item.footprint.Footprint, it's only used if the building definition
        # in the data model doesn't contain building parts, i.e. the building is defined completely
        # by its outline
        self.footprint = None
        self._cache.clear()
        self.assetInfoBldgIndex = None
        self._area = 0.
        # altitude difference for the building footprint projected on the terrain
        self.altitudeDifference = 0.
        
        # attributes from @meta of the style block
        self.buildingUse = None
        self.classifyFacades = 1
    
    def clone(self):
        building = Building()
        return building

    def attr(self, attr):
        return self.outline.tags.get(attr)

    def __getitem__(self, attr):
        """
        That variant of <self.attr(..) is used in a setup script>
        """
        return self.outline.tags.get(attr)
    
    @classmethod
    def getItem(cls, itemFactory, outline, data):
        item = itemFactory.getItem(cls)
        item.init(outline)
        item.data = data
        return item
    
    def setStyleMeta(self, style):
        if style.meta:
            for attr in style.meta.attrs:
                setattr(self, attr, style.meta.attrs[attr])
    
    def area(self):
        if not self._area:
            # remember that <self.footprint> is defined if the building doesn't have parts
            polygon = self.footprint.polygon if self.footprint else self.outlinePolygon
            
            if not polygon.allVerts:
                outline = self.outline
                if outline.t is parse.multipolygon:
                    coords = outline.getOuterData(self.data)
                else:
                    coords = outline.getData(self.data)
                polygon.init( Vector(coord) for coord in coords )
            if polygon.n < 3:
                # the building will be skipped later in method <calculated>
                return 0.
            
            self._area = polygon.area()
        return self._area