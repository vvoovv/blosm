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

import bpy
from renderer import Renderer, Renderer3d
from manager import Manager
from .roof.flat import RoofFlat, RoofFlatMulti
from .roof.pyramidal import RoofPyramidal
from .roof.skillion import RoofSkillion
from .roof.mesh import RoofMesh
from .roof.profile import *
from .roof.hipped import RoofHipped
from .roof.half_hipped import RoofHalfHipped
from util.blender import createDiffuseMaterial
from util import zeroVector

# Python tuples to store some defaults to render walls and roofs of OSM 3D buildings
# Indices to access defaults from Python tuple below
roofIndex = 0
wallIndex = 1
tags = ("roof:colour", "building:colour")
defaultColorNames = ("roof", "wall")
defaultColors = ( (0.29, 0.25, 0.21), (1., 0.5, 0.2) )


class BuildingRenderer(Renderer3d):
    
    def __init__(self, app, layerId):
        super().__init__(app)
        layer = app.getLayer(layerId)
        self.layer = layer
        # set layer offsets <layer.location>, <layer.meshZ> and <layer.parentLocation> to zero
        layer.location = None
        layer.meshZ = 0.
        if layer.parentLocation:
            layer.parentLocation[2] = 0.
        else:
            layer.parentLocation = zeroVector()
        
        if app.terrain:
            # the attribute <singleObject> of the buildings layer doesn't depend on availability of a terrain
            layer.singleObject = app.singleObject
            # the attribute <singleObject> of the buildings layer doesn't depend on availability of a terrain
            layer.layered = app.layered
            # no need to add a SHRINKWRAP modifier for buildings
            layer.swModifier = False
            # no need to slice Blender mesh
            layer.sliceMesh = False
        # create instances of classes that deal with specific roof shapes
        self.flatRoofMulti = RoofFlatMulti()
        self.roofs = {
            'flat': RoofFlat(),
            'gabled': RoofProfile(gabledRoof),
            'pyramidal': RoofPyramidal(),
            'skillion': RoofSkillion(),
            'hipped': RoofHipped(gabledRoof),
            'dome': RoofMesh("roof_dome"),
            'onion': RoofMesh("roof_onion"),
            'round': RoofProfile(roundRoof),
            'half-hipped': RoofHalfHipped(),
            'gambrel': RoofProfile(gambrelRoof),
            'saltbox': RoofProfile(saltboxRoof)
        }
        self.defaultRoof = self.roofs[app.defaultRoofShape]
        
        # set renderer for each roof
        for r in self.roofs:
            self.roofs[r].r = self
        self.flatRoofMulti.r = self
        
        self.defaultMaterialIndices = [None, None]
        # References to Blender materials used by roof Blender meshes
        # loaded from a .blend library file
        self.defaultMaterials = [None, None]
    
    def render(self, building, osm):
        parts = building.parts
        outline = building.element
        app = self.app
        self.parts = parts
        self.outline = outline
        
        if not app.singleObject:
            self.calculateOffset(outline, osm)
        elif app.terrain:
            self.projectOnTerrain(outline, osm)
        
        if app.terrain and not self.offsetZ:
            # The building is located outside the terrain.
            # Perform cleanup and exit
            if not app.singleObject:
                self.offset = None
            return
        
        self.preRender(outline, self.layer)
        
        if parts:
            # reset material indices and Blender materials derived from <outline>
            for i in range(2):
                self.defaultMaterialIndices[i] = None
                self.defaultMaterials[i] = None
        if not parts or outline.tags.get("building:part") == "yes":
            # render building outline
            self.renderElement(outline, building, osm)
        if parts:
            for part in parts:
                self.renderElement(part, building, osm)
        
        # cleanup <self.offset> and <self.offsetZ>
        if not app.singleObject:
            self.offset = None
        if app.terrain and not self.offsetZ is None:
            self.offsetZ = None
        
        self.postRender(outline)
    
    def renderElement(self, element, building, osm):
        # get a class instance created in the constructor to deal with a specific roof shape
        roof = self.roofs.get(element.tags.get("roof:shape"), self.defaultRoof)
        if element.t is Renderer.multipolygon:
            # check if the multipolygon has holes
            if element.hasInner():
                # flat roof is always for multipolygons with holes
                roof = self.flatRoofMulti
                data = None
            else:
                # treat each outer polygon of the multipolygon as a single polygon
                for _l in element.ls:
                    self._renderElement(element, building, roof, element.getLinestringData(_l, osm), osm)
                return
        else:
            data = element.getData(osm)
        self._renderElement(element, building, roof, data, osm)
    
    def _renderElement(self, element, building, roof, data, osm):
        """
        Do actual stuff for <self.renderElement(..) here>
        """
        app = self.app
        z1 = building.getMinHeight(element, app)
        roof.init(element, data, z1, osm)
        if not roof.valid:
            return
        
        roofHeight = roof.getHeight(app)
        z2 = building.getHeight(element)
        if z2 is None:
            # no tag <height> or invalid value
            roofMinHeight = building.getRoofMinHeight(element, app)
            z2 = roofMinHeight + roofHeight
        else:
            roofMinHeight = z2 - roofHeight
        wallHeight = roofMinHeight - z1
        # validity check
        if wallHeight < 0.:
            return
        elif wallHeight < zero:
            # no building walls, just a roof
            wallHeight = None
        
        #print(element.tags["id"]) #DEBUG OSM id
        if roof.make(z2, z1 if wallHeight is None else roofMinHeight, None if wallHeight is None else z1, osm):
            roof.render()

    def getRoofMaterialIndex(self, element):
        """
        Returns the material index for the building roof
        
        Args:
            element: OSM element (building=* or building:part=*)
        """
        return self.getMaterialIndexByPart(element, roofIndex)

    def getWallMaterialIndex(self, element):
        """
        Returns the material index for the building walls
        
        Args:
            element: OSM element (building=* or building:part=*)
        """
        return self.getMaterialIndexByPart(element, wallIndex)
    
    def getMaterialIndexByPart(self, element, partIndex):
        """
        Returns the material index either for building walls or for buildings roof
        
        Args:
            element: OSM element (building=* or building:part=*)
            partIndex (int): Equal to either <roofIndex> or <wallIndex>
        """
        # material name is just the related color (either a hex or a CSS color)
        name = Manager.normalizeColor(element.tags.get(tags[partIndex]))
        
        if name is None:
            # <name> is invalid as a color name
            if self.outline is element:
                # Building oultine is rendererd if there are no parts or
                # if the outline has <building:part=yes>
                
                # take the name for the related default color
                name = defaultColorNames[partIndex]
                # check if Blender material has been already created
                materialIndex = self.getMaterialIndexByName(name)
                if materialIndex is None:
                    # the related Blender material hasn't been created yet, so create it
                    materialIndex = self.getMaterialIndex( createDiffuseMaterial(name, defaultColors[partIndex]) )
                if self.parts:
                    # If there are parts, store <materialIndex>,
                    # since it's set for a building part, if it doesn't have an OSM tag for color
                    self.defaultMaterialIndices[partIndex] = materialIndex
            else:
                # this a building part (building:part=yes), not the building outline
                
                # check if the material index for the default color has been set before
                if self.defaultMaterialIndices[partIndex] is None:
                    # The material index hasn't been set before
                    if self.defaultMaterials[partIndex] is None:
                        # A Blender material also hasn't been set before
                        # Get the material index for the default color,
                        # i.e. the material index for <self.outline>
                        materialIndex = self.getMaterialIndexByPart(self.outline, partIndex)
                    else:
                        # A Blender material also has been set before
                        # Append the Blender material to the related Blender object and
                        # get its material index
                        materialIndex = self.getMaterialIndex(self.defaultMaterials[partIndex])
                        # Store the material index, so we won't need to check
                        # defaultMaterials for the next building part
                        self.defaultMaterialIndices[partIndex] = materialIndex
                else:
                    # The material index for the default color has been set before, so use it,
                    # i.e. use the color for <self.outline>
                    materialIndex = self.defaultMaterialIndices[partIndex]
        else:
            # check if the related Blender material has been already created
            materialIndex = self.getMaterialIndexByName(name)
            if materialIndex is None:
                # The related Blender material hasn't been already created,
                # so create it
                materialIndex = self.getMaterialIndex( createDiffuseMaterial(name, Manager.getColor(name)) )
            # If <element is self.outline> and there are parts, store <materialIndex>,
            # since it's set for a building part, if it doesn't have an OSM tag for color
            if element is self.outline and self.parts:
                self.defaultMaterialIndices[partIndex] = materialIndex
        return materialIndex
    
    def getRoofMaterial(self, element):
        """
        Returns a Blender material for the building roof
        
        Args:
            element: OSM element (building=* or building:part=*)
        """
        # material name is just the related color (either a hex or a CSS color)
        name = Manager.normalizeColor(element.tags.get(tags[roofIndex]))
        
        if name is None:
            # <name> is invalid as a color name
            if self.outline is element:
                # Building oultine is rendererd if there are no parts or
                # if the outline has <building:part=yes>
                
                # take the name for the related default color
                name = defaultColorNames[roofIndex]
                # check if Blender material has been created before
                material = bpy.data.materials.get(name)
                if material is None:
                    # the related Blender material hasn't been created yet, so create it
                    material = createDiffuseMaterial(name, defaultColors[roofIndex])
                if self.parts:
                    # If there are parts, store <material>,
                    # since it will set for the roof of another building part,
                    # if the building part doesn't have an OSM tag for the roof color
                    self.defaultMaterials[roofIndex] = material
            else:
                # this a building part (building:part=yes), not the building outline
                
                # check if the Blender material for the default color has been set before
                if self.defaultMaterials[roofIndex] is None:
                    # Check if the material index for the default color has been set before
                    # Use the related Blender material for that index,
                    # if it has been set before
                    # Otherwise get the roof Blender material for the default color,
                    # i.e. the roof material for <self.outline>
                    material = self.getRoofMaterial(self.outline)\
                        if self.defaultMaterialIndices[roofIndex] is None else\
                        self.obj.data.materials[self.defaultMaterialIndices[roofIndex]]
                else:
                    # The Blender material for the default color has been set before, so use it,
                    # i.e. use the roof color for <self.outline>
                    material = self.defaultMaterials[roofIndex]
        else:
            # check if the related Blender material has been created before
            material = bpy.data.materials.get(name)
            if material is None:
                # The related Blender material hasn't been created before,
                # so create it
                material = createDiffuseMaterial(name, Manager.getColor(name))
            # If <element is self.outline> and there are parts, store <material>,
            # since it will set for the roof of another building part,
            # if the building part doesn't have an OSM tag for the roof color
            if element is self.outline and self.parts:
                self.defaultMaterials[roofIndex] = material
        return material
    
    def calculateOffset(self, outline, osm):
        # take the first vertex of the outline as the offset
        self.offset = Vector(
            next( outline.getOuterData(osm) if outline.t is Renderer.multipolygon else outline.getData(osm) )
        )
        if self.app.terrain:
            self.offsetZ = self.app.terrain.project(self.offset)
    
    def projectOnTerrain(self, outline, osm):
        # take the first vertex of the outline as the offset
        offset = self.app.terrain.project(
            next( outline.getOuterData(osm) if outline.t is Renderer.multipolygon else outline.getData(osm) )
        )
        if offset:
            self.offsetZ = offset[2]
    
    def getVert(self, vert):
        if not self.app.singleObject:
            vert -= self.offset
        elif not self.offsetZ is None:
            vert[2] += self.offsetZ
        return vert