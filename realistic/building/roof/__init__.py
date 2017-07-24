import math
from building.roof import Roof
from manager import Manager
from util.osm import parseNumber


class RoofRealistic:
    
    def init(self, element, data, osm, app):
        super().init(element, data, osm, app)
        # material renderer for walls
        self.mrw = None
        # material renderer for roof
        self.mrr = None
        self._numLevels = None
        self._levelHeights = None
        self._roofColor = None
    
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
    
    @property
    def roofColor(self):
        if self._roofColor is None:
            roofColor = Manager.normalizeColor(self.element.tags.get("roof:colour"))
            self._roofColor = Manager.getColorFromHex(roofColor) if roofColor else 0
        return self._roofColor

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