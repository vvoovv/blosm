from mathutils import Vector
from .roof_flat import RoofFlat
from item.facade import Facade
from item.roof_flat_multi import RoofFlatMulti as ItemRoofFlatMulti
from util.polygon import PolygonCW


class RoofMulti:
    
    def do(self, footprint):
        roofItem = self.init(footprint)
        if footprint.valid:
            if roofItem.innerPolygons:
                if self.renderAfterExtrude:
                    self.render(footprint, roofItem)
                else:
                    self.extrude(footprint, roofItem)
                    footprint.roofItem = roofItem
                    footprint.roofRenderer = self.roofRenderer
            else:
                footprint.element.makePolygon()
                self.volumeAction.volumeGenerators[footprint.getStyleBlockAttr("roofShape")].do(
                    footprint
                )
    
    def extrude(self, footprint, roofItem):
        super().extrude(footprint, roofItem)
        self.extrudeInnerPolygons(footprint, roofItem)
    
    def extrudeInnerPolygons(self, footprint, roofItem):
        #
        # deal with the inner polygons below
        #
        building = footprint.building
        facades = footprint.facades
        verts = building.verts
        indexOffset = len(verts)
        z = footprint.roofVerticalPosition if self.extrudeTillRoof else footprint.height
        
        for polygon in roofItem.innerPolygons:
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
    
    def init(self, footprint):
        data = self.data
        roofItem = super().init(footprint, footprint.element.getOuterData(data))
        if not footprint.valid:
            return
        z1 = footprint.minHeight
        element = footprint.element
        innerPolygons = roofItem.innerPolygons
        
        for _l in element.ls:
            if _l.role is data.outer:
                continue
            # create an inner polygon located at <minHeight>
            innerPolygon = PolygonCW()
            innerPolygon.init( Vector((coord[0], coord[1], z1)) for coord in element.getLinestringData(_l, data) )
            if innerPolygon.numEdges < 3:
                continue
            # check the direction of vertices, it must be clockwise (!)
            innerPolygon.checkDirection()
            innerPolygons.append(innerPolygon)
        return roofItem


class RoofFlatMulti(RoofMulti, RoofFlat):
    
    def __init__(self, data, volumeAction, itemRenderers):
        super().__init__("RoofFlatMulti", data, volumeAction, itemRenderers)
    
    def getRoofItem(self, footprint):
        return ItemRoofFlatMulti.getItem(
            self.itemFactory,
            footprint,
            self.getRoofFirstVertIndex(footprint)
        )