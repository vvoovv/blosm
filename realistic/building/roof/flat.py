import math
from util.osm import parseNumber
from building.roof.flat import Roof, RoofFlat


class RoofFlatRealistic(RoofFlat):
    
    def init(self, element, data, osm, app):
        super().init(element, data, osm, app)
        # material manager
        self.mm = None
        self._numLevels = None
    
    def render(self):
        r = self.r
        if r.bldgPreRender:
            r.bldgPreRender(self)
        super().render()
        # cleanup
        if self.mm:
            self.mm = None
    
    def setMaterialManager(self, constructor):
        # mm stands for "material manager"
        mm = self.r.getMaterialManager(constructor)
        mm.init()
        # set building <b> attribute to <self>
        mm.b = self
        self.mm = mm
    
    def renderWalls(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        materialIndex = r.getWallMaterialIndex(self.element)
        uvLayer = bm.loops.layers.uv[0]
        uvLayerSize = bm.loops.layers.uv[1]
        # create BMesh faces for the building walls
        for f in (bm.faces.new(verts[i] for i in indices) for indices in self.wallIndices):
            w = (f.verts[1].co - f.verts[0].co).length
            h = (f.verts[-1].co - f.verts[0].co).length
            size = (w, h)
            f.loops[0][uvLayer].uv = (0., 0.)
            f.loops[1][uvLayer].uv = (w, 0.)
            f.loops[2][uvLayer].uv = size
            f.loops[3][uvLayer].uv = (0., h)
            for i in range(4):
                f.loops[i][uvLayerSize].uv = size
            if self.mm:
                self.mm.render(f)
            else:
                f.material_index = materialIndex
    
    @property
    def numLevels(self):
        if not self._numLevels:
            tags = self.element.tags
            n = tags.get("building:levels")
            if not n is None:
                n = parseNumber(n)
            if n is None:
                n = self.roofMinHeight/self.r.app.levelHeight + 1 - Roof.groundFloorFactor\
                    if self.z1 else\
                    math.floor((self.roofMinHeight-self.z1)/self.r.app.levelHeight)
            else:
                _n = tags.get("building:min_level")
                if not _n is None:
                    _n = parseNumber(_n)
                    if not _n is None:
                        n -= _n
            self._numLevels = n
        return self._numLevels

    def getOsmMaterial(self):
        element = self.element
        material = element.tags.get("building:material")
        if not material and not element is self.r.outline:
            material = self.r.outline.tags.get("building:material")
        return material