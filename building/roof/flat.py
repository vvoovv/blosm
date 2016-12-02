import bmesh
from mathutils import Vector
from util.osm import parseNumber
from util.polygon import Polygon
from . import Roof


class RoofFlat(Roof):
    
    defaultHeight = 0.5
    
    def init(self, element, osm):
        super().init(element, osm)
        
    def getHeight(self):
        tags = self.element.tags
        return parseNumber(tags["roof:height"], self.defaultHeight)\
            if "roof:height" in tags\
            else self.defaultHeight

    def make(self, bldgMaxHeight, roofMinHeight, bldgMinHeight, osm):
        if bldgMinHeight is None:
            return
        polygon = self.polygon
        # update <polygon.allVerts> with vertices translated along z-axis
        for _i in range(polygon.n):
            i = polygon.indices[_i]
            vert = polygon.allVerts[i].copy()
            vert.z = bldgMaxHeight
            polygon.allVerts[i] = vert
        
        self.sidesIndices = self.polygon.sidesPrism(bldgMinHeight)
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
        bm = r.bm
        verts = tuple( bm.verts.new(v) for v in self.allVerts )
        edges = tuple(
            bm.edges.new( (verts[polygon.indices[i-1]], verts[polygon.indices[i]]) )\
            for polygon in self.polygons\
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
        
        #f = bm.faces.new(verts[i] for i in self.polygon.indices)
        #f.material_index = r.getMaterialIndex(self.element)
        
        #materialIndex = r.getSideMaterialIndex(self.element)
        #for f in (bm.faces.new(verts[i] for i in indices) for indices in sidesIndices):
        #    f.material_index = materialIndex