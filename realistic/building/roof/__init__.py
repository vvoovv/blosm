"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
from building.roof import Roof
from building.roof.mesh import RoofMesh
from manager import Manager
from util.osm import parseNumber
from util import zAxis


class RoofRealistic:
    
    wallHeightWithoutWindows = 1.9
    
    def init(self, element, data, osm, app):
        super().init(element, data, osm, app)
        if not self.valid:
            return
        # material renderer for walls
        self.mrw = None
        # material renderer for roof
        self.mrr = None
        self._numLevels = None
        self._levelHeights = None
        self._wallsColor = None
        self._roofColor = None
        # Does the building or building part have windows?
        self.noWindows = self.noWalls or self.wallHeight < RoofRealistic.wallHeightWithoutWindows
    
    def render(self):
        r = self.r
        if r.bldgPreRender:
            r.bldgPreRender(self, r.app)
        
        mrw = self.mrw
        mrr = self.mrr
        
        if self.mrw:
            mrw.checkBuildingChanged()
        if self.mrr:
            mrr.checkBuildingChanged()
        
        super().render()
        
        # cleanup
        if mrw:
            self.mrw = None
        if mrr:
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
            self.mrr.renderRoof(f)

    def setMaterialWalls(self, name):
        # mrw stands for "material renderer for walls"
        mrw = self.r.getMaterialRenderer(name)
        if mrw:
            mrw.init()
            if mrw.valid:
                # set building <b> attribute to <self>
                mrw.b = self
                self.mrw = mrw
                mrw.isForWalls = True
    
    def setMaterialRoof(self, name):
        # mrr stands for "material renderer for roof"
        mrr = self.r.getMaterialRenderer(name)
        if mrr:
            mrr.init()
            if mrr.valid:
                # set building <b> attribute to <self>
                mrr.b = self
                self.mrr = mrr
                mrr.isForRoof = True

    @property
    def numLevels(self):
        """
        Returns the number of levels in the building or building part.
        
        The number of levels is calculated as the difference
        <building:levels> - <building:min_level>,
        where both elements represent the value of the related tags OR
        the heights in meters. See the code below.
        """
        if not self._numLevels:
            n = self._levels
            if n is None:
                n = self.getLevels(False)
            if n is None:
                # calculate the number of levels out of heights
                n = math.floor(
                    (self.roofVerticalPosition-self.z1)/self.levelHeight\
                    if self.z1 else\
                    self.roofVerticalPosition/self.levelHeight + 1 - Roof.groundLevelFactor
                )
                if not n:
                    n = 1.
            # The condition below means:
            # it doesn't make sense to process <building:min_level> if <n> is equal to 1.
            elif n>1:
                # processing <building:min_level> and <building:min_height>
                _n = self.getMinLevel()
                # the second condition is a sanity check
                if not _n is None and _n < n:
                    n -= _n
                elif self.z1:
                    levelHeight = self.roofVerticalPosition/(Roof.groundLevelFactor - 1 + n)
                    _n = math.floor(self.z1/levelHeight + 1 - Roof.groundLevelFactor)
                    if _n:
                        n -= _n
                if not n:
                    n = 1.
            self._numLevels = n
        return self._numLevels
    
    @property
    def levelHeights(self):
        if not self._levelHeights:
            h = self._levelHeight
            if h is None:
                h = (self.roofVerticalPosition - self.z1)/self.numLevels\
                    if self.z1 else\
                    self.roofVerticalPosition/(Roof.groundLevelFactor + self.numLevels - 1)
            self._levelHeights = (h, Roof.groundLevelFactor*h)
        return self._levelHeights

    @property
    def wallsColor(self):
        if self._wallsColor is None:
            wallsColor = Manager.normalizeColor( self.getOsmTagValue("building:colour") )
            self._wallsColor = Manager.getColor(wallsColor) if wallsColor else 0
        return self._wallsColor
    
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