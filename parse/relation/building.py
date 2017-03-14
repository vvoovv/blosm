"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
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

from parse import Osm
from . import Relation
from .multipolygon import Multipolygon


class Building(Relation):
    
    # use __slots__ for memory optimization
    __slots__ = ("outline",)
    
    def __init__(self, osm):
        if osm.app.mode == "2D":
            # OSM relation of the type 'building' makes sense only in the 3D mode
            self.valid = False
            return
        super().__init__()
    
    def preprocess(self, members, osm):
        outline = None
        # the first pass: looking for a relation member with the role 'outline'
        for mType, mId, mRole in members:
            if mRole is Osm.outline:
                if mType is Osm.way:
                    # all OSM ways are already available
                    if mId in osm.ways:
                        # Check if the candidate for a building outline has <b> attribute,
                        # which points to an instance of <building.manager.Building>
                        if not hasattr(osm.ways[mId], "b"):
                            # <outline> can't serve as a building outline
                            break
                        outline = (mId, Osm.way)
                        break
                elif mType is Osm.relation:
                    # get either an existing relation encountered before or create an empty relation
                    outline = osm.getRelation(mId, Multipolygon)
                    if outline.tags:
                        # Check if the candidate for a building outline has <b> attribute,
                        # which points to an instance of <building.manager.Building>
                        if not hasattr(outline, "b"):
                            # <outline> can't serve as a building outline
                            outline = None
                            break
                    else:
                        # <not outline.tags> means that outline is an empty relation,
                        # i.e. not encountered in the OSM file yet.
                        # Ensure that the OSM relation serving as outline has <b> attribute,
                        # which will be needed later;
                        # the attribute <b> is used to store an instance of <building.manager.Building>
                        outline.b = None
                    outline = (mId, Osm.relation)
                    break
        if outline:
            self.outline = outline
        else:
            # A relation member with the role 'outline' not found or
            # the relation member can't serve as a building outline
            self.valid = False
            return
    
    def process(self, members, tags, osm):
        # the first pass
        self.preprocess(members, osm)
        if not self.valid:
            # return True since the relation shouldn't be added to <app.incompleteRelations>
            return True
        # The second pass:
        # For all building parts from <self.members> set attribute <o> to <self.outline>
        for mType, mId, mRole in members:
            if mRole is Osm.part:
                if mType is Osm.way:
                    # all OSM ways are already available
                    if mId in osm.ways:
                        osm.ways[mId].o = self.outline
                elif mType is Osm.relation:
                    osm.getRelation(mId, Multipolygon).o = self.outline
        # return True since the relation shouldn't be added to <app.incompleteRelations>
        return True