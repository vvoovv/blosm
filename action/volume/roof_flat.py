from .roof import Roof
from item.facade import Facade
from item.roof_flat import RoofFlat as ItemRoofFlat
from .geometry.rectangle import Rectangle
from mathutils import Vector


class RoofFlat(Roof):
    
    # default roof height
    height = 1.
    
    def __init__(self, data, itemStore, itemFactory, roofRenderer):
        super().__init__(data, itemStore, itemFactory)
        self.roofRenderer = roofRenderer
        self.rectangleGeometry = Rectangle()
        self.hasRoofLevels = False
        self.extrudeTillRoof = False
    
    def render(self, footprint, facadeRenderer):
        # <firstVertIndex> is the index of the first vertex of the polygon that defines the roof base;
        # since it depends on the total number of building vertices, we calculated it before any operation
        # that creates building geometry 
        firstVertIndex = len(footprint.building.verts) + footprint.polygon.n
        
        self.extrude(footprint)
        facadeRenderer.render(footprint)
        self.roofRenderer.render(
            self.getRoofItem(
                footprint,
                firstVertIndex
            )
        )
    
    def getRoofItem(self, footprint, firstVertIndex):
        return ItemRoofFlat.getItem(
            self.itemFactory,
            footprint,
            firstVertIndex
        )
    
    def extrude(self, footprint):
        building = footprint.building
        facades = footprint.facades
        verts = building.verts
        indexOffset = len(verts)
        polygon = footprint.polygon
        numVerts = polygon.n
        
        # create vertices
        z = footprint.roofVerticalPosition if self.extrudeTillRoof else footprint.height
        # verts for the lower cap
        verts.extend(v for v in polygon.verts)
        # verts for the upper cap
        verts.extend(Vector((v.x, v.y, z)) for v in polygon.verts)
        
        # the starting side
        _in = indexOffset+numVerts
        facades.append(Facade.getItem(
            self.itemFactory,
            footprint,
            self.rectangleGeometry,
            (_in-1, indexOffset, _in, _in+numVerts-1),
            self.rectangleGeometry.getUvs(
                (verts[indexOffset] - verts[_in-1]).length,
                footprint.wallHeight
            )
        ))
        # the rest of the sides
        facades.extend(
            Facade.getItem(
                self.itemFactory,
                footprint,
                self.rectangleGeometry,
                (indexOffset+i-1, indexOffset+i, _in+i, _in+i-1),
                self.rectangleGeometry.getUvs(
                    (verts[indexOffset+i] - verts[indexOffset+i-1]).length,
                    footprint.wallHeight
                )
            ) for i in range(1, numVerts)
        )