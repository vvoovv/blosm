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

from building.renderer import *

from .roof.flat import RoofFlatRealistic, RoofFlatMultiRealistic
from .roof.profile import RoofProfileRealistic
from .roof.pyramidal import RoofPyramidalRealistic
from .roof.skillion import RoofSkillionRealistic
from .roof.hipped import RoofHippedRealistic
from .roof.half_hipped import RoofHalfHippedRealistic
from .roof.mansard import RoofMansardRealistic
from .roof.mesh import RoofMeshRealistic


class RealisticBuildingRenderer(BuildingRenderer):
    
    def __init__(self, app, **kwargs):
        self.bldgPreRender = None
        super().__init__(app)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        # A Python dictionary for mapping between material names and material renderers;
        # it's normally set via <kwargs>
        if not self.materials:
            self.materials = {}
        # a Python dictionary of material renderers
        self.materialRenderers = {}
        # a Python set of names of processed material groups;
        # it maybe used by several material renderers
        self.materialGroups = set()
    
    def getMaterialRenderer(self, name):
        # <mr> stands for "material renderer"
        mr = self.materialRenderers.get(name)
        if not mr:
            # <md> stands for material definition
            md = self.materials.get(name)
            if md:
                # <md[0]> is a constructor
                mr = md[0](self, name, *md[1:])\
                    if isinstance(md, tuple) else\
                    md(self, name) # <md> is just a constructor
                self.materialRenderers[name] = mr
        return mr
    
    def initRoofs(self):
        """
        The override of the parent class method
        """
        self.flatRoofMulti = RoofFlatMultiRealistic()
        self.roofs = {
            'flat': RoofFlatRealistic(),
            'gabled': RoofProfileRealistic(gabledRoof),
            'pyramidal': RoofPyramidalRealistic(),
            'skillion': RoofSkillionRealistic(),
            'hipped': RoofHippedRealistic(),
            'dome': RoofMeshRealistic("roof_dome"),
            'onion': RoofMeshRealistic("roof_onion"),
            'round': RoofProfileRealistic(roundRoof),
            'half-hipped': RoofHalfHippedRealistic(),
            'gambrel': RoofProfileRealistic(gambrelRoof),
            'saltbox': RoofProfileRealistic(saltboxRoof),
            'mansard': RoofMansardRealistic()
        }