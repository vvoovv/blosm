from mathutils import Vector
from .roof_flat import RoofFlat
from item.facade import Facade
from item.roof_flat_multi import RoofFlatMulti as ItemRoofFlatMulti
from util.polygon import PolygonCW


class RoofFlatMulti(RoofFlat):
    
    def __init__(self, data, itemStore, itemFactory, facadeRender, roofRenderer):
        super().__init__(data, itemStore, itemFactory, facadeRender, roofRenderer)
        self.innerPolygons = []

    def do(self, footprint):
        self.init(footprint)
        if footprint.valid:
            self.render(footprint)
    
    def getRoofItem(self, footprint, firstVertIndex):
        item = ItemRoofFlatMulti.getItem(
            self.itemFactory,
            footprint,
            firstVertIndex
        )
        item.innerPolygons = self.innerPolygons
        return item
    
    def init(self, footprint):
        data = self.data
        super().init(footprint, footprint.element.getOuterData(data))
        z1 = footprint.minHeight
        element = footprint.element
        innerPolygons = self.innerPolygons
        innerPolygons.clear()
        
        for _l in element.ls:
            if _l.role is data.outer:
                continue
            # create an inner polygon located at <minHeight>
            innerPolygon = PolygonCW()
            innerPolygon.init( Vector((coord[0], coord[1], z1)) for coord in element.getLinestringData(_l, data) )
            if innerPolygon.n < 3:
                continue
            # check the direction of vertices, it must be clockwise (!)
            innerPolygon.checkDirection()
            innerPolygons.append(innerPolygon)
    
    def extrude(self, footprint):
        super().extrude(footprint)
        
        #
        # deal with the inner polygons below
        #
        building = footprint.building
        facades = footprint.facades
        verts = building.verts
        indexOffset = len(verts)
        z = footprint.roofVerticalPosition if self.extrudeTillRoof else footprint.height
        
        for polygon in self.innerPolygons:
            numVerts = polygon.n
            
            # create vertices
            # verts for the lower cap
            verts.extend(v for v in polygon.verts)
            # verts for the upper cap
            verts.extend(Vector((v.x, v.y, z)) for v in polygon.verts)
            
            # the starting side
            _in = indexOffset+numVerts
            facades.append(Facade.getItem(
                self,
                footprint,
                (_in-1, indexOffset, _in, _in+numVerts-1),
                0 # edge index
            ))
            # the rest of the sides
            facades.extend(
                Facade.getItem(
                    self,
                    footprint,
                    (indexOffset+i-1, indexOffset+i, _in+i, _in+i-1),
                    i # edge index
                ) for i in range(1, numVerts)
            )
            # mark the created facades as inner
            for i in range(-numVerts, 0):
                facades[i].outer = False
                facades[i].normal.negate()
            
            indexOffset += 2*numVerts