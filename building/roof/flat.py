import bmesh
from mathutils import Vector
from util.polygon import Polygon
from . import Roof


class RoofFlat(Roof):
    
    defaultHeight = 0.5
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        if bldgMinHeight is None:
            return
        polygon = self.polygon
        n = len(self.verts)
        
        self.polygon.sidesPrism(bldgMaxHeight, self.wallIndices)
        self.roofIndices.append( tuple(range(n, n+polygon.n)) )
        return True


class RoofFlatMulti(RoofFlat):
    
    def init(self, element, osm):
        self.element = element
    
    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        if bldgMinHeight is None:
            return
        
        element = self.element
        
        verts = []
        polygons = []
        indexOffset = 0
        for _l in element.l:
            verts.extend(
                Vector((v.x, v.y, bldgMaxHeight)) for v in element.getLinestringData(_l, osm)
            )
            n = len(verts)
            _n = n - indexOffset
            polygons.append(
                Polygon(
                    tuple(range(indexOffset, n)),
                    verts
                )
            )
            indexOffset = n
        # vertices for the bottom
        verts.extend(Vector((verts[i].x, verts[i].y, bldgMinHeight)) for i in range(n))
        self.polygons = polygons
        self.allVerts = verts
        return True
    
    def render(self, r):
        element = self.element
        polygons = self.polygons
        bm = r.bm
        verts = tuple( bm.verts.new(v) for v in self.allVerts )
        edges = tuple(
            bm.edges.new( (verts[polygon.indices[i-1]], verts[polygon.indices[i]]) )\
            for polygon in polygons\
            for i in range(polygon.n)
        )
        
        # a magic function that does everything
        geom = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)
        # check the normal direction of the created faces and assign material to all BMFace
        materialIndex = r.getMaterialIndex(element)
        for f in geom["geom"]:
            if isinstance(f, bmesh.types.BMFace):
                if f.normal.z < 0.:
                    f.normal_flip()
                f.material_index = materialIndex
        
        indexOffset1 = 0
        indexOffset2 = len(verts)//2
        sidesIndices = []
        for polygon in polygons:
            n = polygon.n
            # the first edge of the polygon
            edge = edges[indexOffset1]
            if not edge.link_loops:
                # something wrong with the topology of the related OSM multipolygon
                # update index offsets to switch to the next polygon (i.e. a closed linestring)
                indexOffset1 += n
                indexOffset2 += n
                # skip that polygon
                continue
            # a BMLoop for <edge>
            l = edge.link_loops[0]
            keepDirection = l.link_loop_next.vert == verts[indexOffset1]
            
            sidesIndices.extend(
                (
                    indexOffset1 - 1 + (i if i else n),
                    indexOffset2 - 1 + (i if i else n),
                    indexOffset2 + i, 
                    indexOffset1 + i
                )\
                if keepDirection else\
                (
                    indexOffset1 - 1 + (i if i else n),
                    indexOffset1 + i,
                    indexOffset2 + i,
                    indexOffset2 - 1 + (i if i else n)
                )\
                for i in range(n)
            )
            
            indexOffset1 += n
            indexOffset2 += n
        
        materialIndex = r.getSideMaterialIndex(element)
        for f in (bm.faces.new(verts[i] for i in indices) for indices in sidesIndices):
            f.material_index = materialIndex