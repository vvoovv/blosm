from util import zAxis
from . import Roof


class RoofPyramidal(Roof):
    
    defaultHeight = 0.5
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        verts = self.verts
        roofIndices = self.roofIndices
        indices = polygon.indices
        
        if not bldgMinHeight is None:
            indexOffset = len(verts)
            polygon.sidesPrism(roofMinHeight, self.wallIndices)
        
        topIndex = len(verts)
        verts.append(
            polygon.center + (bldgMaxHeight - (roofMinHeight if bldgMinHeight is None else bldgMinHeight)) * zAxis
        )
        
        if bldgMinHeight is None:
            roofIndices.extend(
                (indices[i-1], indices[i], topIndex) for i in range(polygon.n)
            )
        else:
            # the starting triangle
            roofIndices.append((indexOffset + polygon.n - 1, indexOffset, topIndex))
            roofIndices.extend(
                (i - 1, i, topIndex) for i in range(indexOffset + 1, indexOffset + polygon.n)
            )
            
        return True