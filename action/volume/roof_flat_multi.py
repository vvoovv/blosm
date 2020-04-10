from mathutils import Vector
from .roof_flat import RoofFlat
from util.polygon import PolygonCW


class RoofFlatMulti(RoofFlat):
    
    def __init__(self, data, itemStore, itemFactory, facadeRender, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRender, roofRenderer)
        self.innerPolygons = []

    def do(self, footprint):
        self.init(footprint)
        if footprint.valid:
            self.render(footprint)
    
    def init(self, footprint):
        data = self.data
        super().init(footprint, footprint.element.getOuterData(data))
        z1 = footprint.minHeight
        element = footprint.element
        innerPolygons = self.innerPolygons
        innerPolygons.clear()
        
        for _l in element.ls:
            if _l.role is data.inner:
                continue
            # create an inner polygon located at <minHeight>
            innerPolygon = PolygonCW()
            innerPolygon.init( Vector((coord[0], coord[1], z1)) for coord in element.getLinestringData(_l, data) )
            if innerPolygon.n < 3:
                continue
            # check the direction of vertices, it must be clockwise (!)
            innerPolygon.checkDirection()
            innerPolygons.append(innerPolygon)