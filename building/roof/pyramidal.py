from util import zAxis
from . import Roof


class RoofPyramidal(Roof):
    """
    A Blender object to deal with buildings or building part with a pyramidal roof
    """
    
    defaultHeight = 0.5
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        polygon = self.polygon
        verts = self.verts
        roofIndices = self.roofIndices
        indices = polygon.indices
        
        if not bldgMinHeight is None:
            indexOffset = len(verts)
            # Create sides for the prism with the height <roofMinHeight - bldgMinHeight>,
            # that is based on the <polygon>
            polygon.sidesPrism(roofMinHeight, self.wallIndices)
        
        # index for the top vertex
        topIndex = len(verts)
        verts.append(
            polygon.center + (bldgMaxHeight - (roofMinHeight if bldgMinHeight is None else bldgMinHeight)) * zAxis
        )
        
        # indices for triangles that form the pyramidal roof
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