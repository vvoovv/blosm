from .roof import Roof
from item.facade import Facade
from mathutils import Vector


class RoofFlat(Roof):
    
    # default roof height
    height = 1.
    
    def __init__(self, data, itemStore, itemFactory):
        super().__init__(data, itemStore, itemFactory)
        self.hasRoofLevels = False
    
    def render(self, footprint, building, renderer):
        style = footprint.calculatedStyle
        self.extrude(footprint, building, renderer)
    
    def make(self, footprint, style, building):
        verts = building.verts
        #Facade.getItem(self.itemFactory, part)
        #for edge in footprint.polygon:
        #    polygon = self.polygon
        n = len(self.verts)
        
        # Extrude <polygon> in the direction of <z> axis to bring
        # the extruded part to the height <bldgMaxHeight>
        #polygon.extrude(self.z2, self.wallIndices)
        # fill the extruded part
        #self.roofIndices.append( tuple(range(n, n+polygon.n)) )
        return True
    
    def extrude(self, footprint, building, renderer):
        verts = building.verts
        indexOffset = len(verts)
        _indices = self.indices
        # verts
        z = footprint.height
        # verts for the lower cap
        verts.extend(v for v in footprint.polygon.verts)
        # verts for the upper cap
        verts.extend(Vector((v.x, v.y, z)) for v in verts)
        # the starting side
        #indices.append((_indices[-1], _indices[0], indexOffset, indexOffset + self.n - 1))
        #indices.extend(
        #    (_indices[i-1], _indices[i], indexOffset + i, indexOffset + i - 1) for i in range(1, self.n)
        #)
        