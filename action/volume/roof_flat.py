from .roof import Roof
from item.facade import Facade
from item_renderer.geometry.rectangle import Rectangle
from mathutils import Vector


class RoofFlat(Roof):
    
    # default roof height
    height = 1.
    
    def __init__(self, data, itemStore, itemFactory):
        super().__init__(data, itemStore, itemFactory)
        self.hasRoofLevels = False
        self.rectangleGeometry = Rectangle()
    
    def render(self, footprint, facadeRenderer):
        self.extrude(footprint)
        facadeRenderer.render(footprint)
    
    def extrude(self, footprint):
        building = footprint.building
        facades = footprint.facades
        verts = building.verts
        indexOffset = len(verts)
        polygon = footprint.polygon
        numVerts = polygon.n
        
        # create vertices
        z = footprint.height
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
            (verts[indexOffset] - verts[_in-1]).length,
            footprint.wallHeight,
            0
        ))
        # the rest of the sides
        facades.extend(
            Facade.getItem(
                self.itemFactory,
                footprint,
                self.rectangleGeometry,
                (indexOffset+i-1, indexOffset+i, _in+i, _in+i-1),
                (verts[indexOffset+i] - verts[indexOffset+i-1]).length,
                footprint.wallHeight,
                0
            ) for i in range(1, numVerts)
        )