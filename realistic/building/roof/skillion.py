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

from . import RoofRealistic
from building.roof.skillion import RoofSkillion


class RoofSkillionRealistic(RoofRealistic, RoofSkillion):
    
    def renderWalls(self):
        if self.mrw:
            bm = self.r.bm
            verts = self.verts
            uvLayer = bm.loops.layers.uv[0]
            uvLayerSize = bm.loops.layers.uv[1]
            # The variable <firstFace> is used if there are wall faces (exactly two!)
            # composed of 3 vertices
            firstFace = True
            # create BMesh faces for the building walls
            for f in (bm.faces.new(verts[i] for i in indices) for indices in self.wallIndices):
                origin = f.verts[0].co
                originZ = origin[2]
                w = (f.verts[1].co - origin).length
                size = None if self.noWalls else (w, self.wallHeight)
                f.loops[0][uvLayer].uv = (0., 0.)
                f.loops[1][uvLayer].uv = (w, 0.)
                if len(f.verts) == 4:
                    f.loops[2][uvLayer].uv = (w, f.verts[2].co[2] - originZ)
                    f.loops[3][uvLayer].uv = (0., f.verts[3].co[2] - originZ)
                else: # len(f.verts) == 3
                    f.loops[2][uvLayer].uv = (w if firstFace else 0., f.verts[2].co[2] - originZ)
                    firstFace = False
                if size:
                    for l in f.loops:
                        l[uvLayerSize].uv = size
                self.mrw.renderWalls(f, w)
        else:
            RoofSkillion.renderWalls(self)