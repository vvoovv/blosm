import math
from building.roof import Roof
from building.roof.mesh import RoofMesh
from manager import Manager
from util.osm import parseNumber
from util import zAxis


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

    def renderRoof(self):
        if self.mrr:
            self.renderRoofTextured()
        else:
            super().renderRoof()
    
    def renderRoofTextured(self):
        r = self.r
        bm = r.bm
        verts = self.verts
        uvLayer = bm.loops.layers.uv[0]
        for indices in self.roofIndices:
            # create a BMesh face for the building roof
            f = bm.faces.new(verts[i] for i in indices)
            # The roof face <f> can be concave, so we have to use the normal
            # calculated by BMesh module
            f.normal_update()
            # Find the vertex for newly created roof face <f> with the minimun <z>-coordinate;
            # it will serve as an origin
            minIndex = min(indices, key = lambda i: verts[i].co[2])
            origin = verts[minIndex].co
            # Unit vector along the intersection between the plane of the roof face <f>
            # and horizontal plane. It serves as u-axis for the UV-mapping
            uVec = zAxis.cross(f.normal).normalized()
            for i,roofIndex in enumerate(indices):
                if roofIndex == minIndex:
                    # Some optimization, i.e. no need to perform calculations
                    # as in the case <roofIndex != minIndex>
                    u = 0.
                    v = 0.
                else:
                    # u-coordinate is just a projecton of <verts[roofIndex].co - origin>
                    # on the vector <uVec>
                    u = (verts[roofIndex].co - origin).dot(uVec)
                    v = (verts[roofIndex].co - origin - u*uVec).length
                f.loops[i][uvLayer].uv = (u, v)
            self.mrr.render(f)

    def setMaterialWalls(self, name):
        # mrw stands for "material renderer for walls"
        mrw = self.r.getMaterialRenderer(name)
        mrw.init()
        # set building <b> attribute to <self>
        mrw.b = self
        self.mrw = mrw
    
    def setMaterialRoof(self, name):
        # mrr stands for "material renderer for roof"
        mrr = self.r.getMaterialRenderer(name)
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
            roofColor = Manager.normalizeColor( self.getOsmTagValue("roof:colour") )
            self._roofColor = Manager.getColor(roofColor) if roofColor else 0
        return self._roofColor

    def getOsmTagValue(self, tag):
        """
        Returns a value of the given OSM <tag>, taking into account,
        that the tag can be set either at the building part or at the building outline
        """
        element = self.element
        value = element.tags.get(tag)
        if not value and not element is self.r.outline:
            value = self.r.outline.tags.get(tag)
        return value
    
    @property
    def wallsMaterial(self):
        return self.getOsmTagValue("building:material")
    
    @property
    def roofMaterial(self):
        return self.getOsmTagValue("roof:material")