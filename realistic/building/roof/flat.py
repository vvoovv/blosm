import math
from util.osm import parseNumber
from building.roof.flat import Roof, RoofFlat


class RoofFlatRealistic(RoofFlat):
    
    def init(self, element, data, osm, app):
        super().init(element, data, osm, app)
        # material renderer for walls
        self.mrw = None
        # material renderer for roof
        self.mrr = None
        self._numLevels = None
        self._levelHeights = None
    
    def render(self):
        r = self.r
        if r.bldgPreRender:
            r.bldgPreRender(self)
        super().render()
        # cleanup
        if self.mrw:
            self.mrw = None
        if self.mrr:
            self.mrr = None
    
    def setMaterialRendererW(self, constructor):
        # mrw stands for "material renderer for walls"
        mrw = self.r.getMaterialRenderer(constructor)
        mrw.init()
        # set building <b> attribute to <self>
        mrw.b = self
        self.mrw = mrw
    
    def setMaterialRendererR(self, constructor):
        # mrr stands for "material renderer for roof"
        mrr = self.r.getMaterialRenderer(constructor)
        mrr.init()
        # set building <b> attribute to <self>
        mrr.b = self
        self.mrr = mrr
    
    def renderWalls(self):
        if self.mrw:
            r = self.r
            bm = r.bm
            verts = self.verts
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
                self.mrw.render(f)
        else:
            super().renderWalls()
    
    def renderRoof(self):
        if self.mrr:
            r = self.r
            bm = r.bm
            verts = self.verts
            uvLayer = bm.loops.layers.uv[0]
            # create BMesh faces for the building roof
            for f in (bm.faces.new(verts[i] for i in indices) for indices in self.roofIndices):
                loops = f.loops
                offset = loops[0].vert.co
                loops[0][uvLayer].uv = (0., 0.)
                for i in range(1, len(loops)):
                    loops[i][uvLayer].uv = (loops[i].vert.co - offset)[:2]
                self.mrr.render(f)
        else:
            super().renderRoof()
    
    @property
    def numLevels(self):
        if not self._numLevels:
            tags = self.element.tags
            n = tags.get("building:levels")
            if not n is None:
                n = parseNumber(n)
            if n is None:
                n = math.floor(
                    (self.roofMinHeight-self.z1)/self.r.app.levelHeight\
                    if self.z1 else\
                    self.roofMinHeight/self.r.app.levelHeight + 1 - Roof.groundLevelFactor
                )
            else:
                _n = tags.get("building:min_level")
                if not _n is None:
                    _n = parseNumber(_n)
                    if not _n is None:
                        n -= _n
            self._numLevels = n
        return self._numLevels
    
    @property
    def levelHeights(self):
        if not self._levelHeights:
            h = self.roofMinHeight/(Roof.groundLevelFactor + self.numLevels - 1)
            self._levelHeights = (Roof.groundLevelFactor*h, h)
        return self._levelHeights

    def getOsmMaterial(self, tag):
        element = self.element
        material = element.tags.get(tag)
        if not material and not element is self.r.outline:
            material = self.r.outline.tags.get(tag)
        return material
    
    def getOsmMaterialWalls(self):
        return self.getOsmMaterial("building:material")
    
    def getOsmMaterialRoof(self):
        return self.getOsmMaterial("roof:material")