from util import zAxis
from . import Roof


class RoofPyramidal(Roof):
    
    defaultHeight = 0.5
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        verts = polygon.allVerts
        indices = polygon.indices
        indexOffset = len(verts)
        verts.append(polygon.center + bldgMaxHeight * zAxis)
        # update <polygon.allVerts> with vertices translated along z-axis
        for i in indices:
            vert = polygon.allVerts[i].copy()
            vert.z = roofMinHeight
            verts[i] = vert
        
        self.roofIndices = [(indices[i-1], indices[i], indexOffset) for i in range(polygon.n)]
        self.sidesIndices = None if bldgMinHeight is None else polygon.sidesPrism(bldgMinHeight)
        return True
    
    def render(self, r):
        bm = r.bm
        verts = [bm.verts.new(v) for v in self.polygon.allVerts]
        
        materialIndex = r.getMaterialIndex(self.element)
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.roofIndices):
            f.material_index = materialIndex
        
        if self.sidesIndices:
            materialIndex = r.getSideMaterialIndex(self.element)
            for f in (bm.faces.new(verts[i] for i in indices) for indices in self.sidesIndices):
                f.material_index = materialIndex