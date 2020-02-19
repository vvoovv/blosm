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
import bpy
from util.blender import loadMaterialsFromFile
from util.blender_extra.material import setCustomNodeValue

_isBlender280 = bpy.app.version[1] >= 80


_colors = (
    (0.502, 0., 0.502), # purple
    (0.604, 0.804, 0.196), # yellowgreen
    (0.529, 0.808, 0.922), # skyblue
    (0.565, 0.933, 0.565), # lightgreen
    (0.855, 0.647, 0.125) # goldenrod
)


class UvOnly:
    
    def __init__(self, renderer, baseMaterialName):
        self.valid = True
    
    def renderWalls(self, face, width):
        return
    
    def renderRoof(self, face):
        return
    
    def renderForObject(self, obj, slot):
        return
    
    def init(self):
        return
    
    def checkBuildingChanged(self):
        return


class MaterialRenderer:
    
    vertexColorLayer = "Col"
    
    # default roof color used by some material renderers
    roofColor = (0.29, 0.25, 0.21)
    
    def __init__(self, renderer, baseMaterialName, colors = None):
        self.valid = True
        self.r = renderer
        # base name for Blender materials
        self.materialName = baseMaterialName
        # index of the Blender material to set to a BMFace
        self.materialIndex = -1
        # keep track of building outline
        self.outline = None
        # a data structure to store names of Blender materials
        self.materials = None
        # Do we have multiple groups of materials
        # (e.g. apartments and apartments_with_ground_level)?
        self.multipleGroups = False
        # variables for the default colors
        self.colorIndex = -1
        # a list of colors used for walls or for a roof if a colors wasn't set in OSM
        self.colors = colors
        self.numColors = len(colors) if colors else 0
        # We need to distinguish, if we are dealing with the material
        # for walls or for a roof. That will be set after creating
        # an instance of <MaterialRenderer>
        self.isForWalls = False
        self.isForRoof = False
    
    def requireUvLayer(self, name):
        uv = self.r.bm.loops.layers.uv
        # create a data UV layer
        if not name in uv:
            uv.new(name)
    
    def requireVertexColorLayer(self, name):
        vertex_colors = self.r.bm.loops.layers.color
        # create a vertex color layer for data
        if not name in vertex_colors:
            vertex_colors.new(name)
    
    def setData(self, face, layerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, 0.)
        uvLayer = self.r.bm.loops.layers.uv[layerName]
        for loop in face.loops:
            loop[uvLayer].uv = uv

    def setDataForObject(self, obj, layerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, 0.)
        uvLayer = obj.data.uv_layers[layerName]
        for d in uvLayer.data:
            d.uv = uv
    
    def setColor(self, face, layerName, color):
        if not color:
            color = self.colors[self.colorIndex]
        vertexColorLayer = self.r.bm.loops.layers.color[layerName]
        for loop in face.loops:
            loop[vertexColorLayer] = (color[0], color[1], color[2], 1.) if _isBlender280 else color
    
    def setColorForObject(self, obj, layerName, color):
        if not color:
            color = self.colors[self.colorIndex]
        vertexColorLayer = obj.data.vertex_colors[layerName]
        for d in vertexColorLayer.data:
            d.color = (color[0], color[1], color[2], 1.) if _isBlender280 else color
    
    def setupMaterial(self, name):
        """
        Unlike <self.setupMaterials(..)> the method setups a single Blender material
        with the name <name>
        """
        if not name in bpy.data.materials:
            if not loadMaterialsFromFile(self.r.app.bldgMaterialsFilepath, True, name)[0]:
                self.valid = False
                print("The material %s doesn't exist!" % name)
    
    def setupMaterials(self, groupName, numMaterials=20):
        if groupName in self.r.materialGroups:
            return
        # names of materials to load from a .blend file
        materialsToLoad = []
        # names of materials available after scanning everything
        materials = []
        
        for name in ("%s.%s" % (groupName, i) for i in range(1, numMaterials+1)):
            if name in bpy.data.materials:
                materials.append(name)
            else:
                materialsToLoad.append(name)
        if materialsToLoad:
            loadedMaterials = loadMaterialsFromFile(
                self.r.app.bldgMaterialsFilepath,
                True,
                *materialsToLoad
            )
            materials.extend(m.name for m in loadedMaterials if not m is None)
        
        if materials:
            for name in materials:
                self.initMaterial(name)
            
            # set the number of materials
            numMaterials = len(materials)
            if not self.multipleGroups or numMaterials < self.numMaterials:
                self.numMaterials = numMaterials
            
            # data structure to store names of Blender materials
            _materials = self.materials
            if _materials:
                if not self.multipleGroups:
                    self.multipleGroups = True
                    # create a more complex data structure (a Python dictionary of Python lists)
                    _materials = {}
                    _materials[self.groupName] = self.materials
                    self.materials = _materials
                _materials[groupName] = materials
            else:
                # The name of the first group of Blender materials;
                # <self.groupName> will be used in the code if another group is added
                self.groupName = groupName
                self.materials = materials
        else:
            self.valid = False
            print("No materials with the base name %s have been found" % groupName)
        
        self.r.materialGroups.add(groupName)
    
    def initMaterial(self, name):
        """
        Do something with the Blender material with the <name> marked in <self.setupMaterials(..)>.
        
        The function can be overriden by a child class
        """
        pass
    
    def setMaterial(self, face, groupName=None):
        """
        Set material (actually material index) for the given <face>.
        """
        r = self.r
        materialIndices = r.materialIndices
        materials = r.obj.data.materials
        
        name = self.getMaterialName(groupName)
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
    
    def getMaterial(self, groupName=None):
        return bpy.data.materials[ self.getMaterialName(groupName) ]
    
    def getMaterialName(self, groupName):
        # the name of the current material
        return (
            self.materials[groupName if groupName else self.groupName]\
            if self.multipleGroups else\
            self.materials
        )[self.materialIndex]
    
    def checkBuildingChanged(self):
        r = self.r
        if not self.outline is r.outline:
            self.outline = r.outline
            self.onBuildingChanged()
    
    def onBuildingChanged(self):
        self.updateMaterialIndex()
        self.updateColorIndex()
    
    def updateMaterialIndex(self):
        # increase <self.materialIndex> to use the next material
        self.materialIndex += 1
        if self.materialIndex == self.numMaterials:
            # all available materials have been used, so set <self.materialIndex> to zero
            self.materialIndex = 0
    
    def updateColorIndex(self):
        if (self.isForWalls and not self.b.wallsColor) or (self.isForRoof and not self.b.roofColor):
            # Increase <self.colorIndex> to use it the next time when color for the walls
            # wasn't set
            self.colorIndex += 1
            if self.colorIndex == self.numColors:
                # all available colors have been used, so set <self.colorIndex> to zero
                self.colorIndex = 0
    
    def setSingleMaterial(self, face, name):
        r = self.r
        materialIndices = r.materialIndices
        materials = r.obj.data.materials
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
    
    def getSingleMaterial(self):
        return bpy.data.materials[self.materialName]


class FacadeWithColor(MaterialRenderer):
    
    def __init__(self, renderer, baseMaterialName, colors = None):
        super().__init__(renderer, baseMaterialName,colors)
        if self.r.app.litWindows:
            self.materialName += "_emission"
            self.materialName2 = "%s_ground_level_emission" % baseMaterialName
        else:
            self.materialName2 = "%s_ground_level" % baseMaterialName

    def initMaterial(self, name):
        app = self.r.app
        if app.litWindows:
            FacadeWithColor.setWindowEmissionRatio(bpy.data.materials[name], app.litWindows)

    @staticmethod
    def updateLitWindows(addon, context):
        percentage = addon.litWindows
        for m in bpy.data.materials:
            FacadeWithColor.setWindowEmissionRatio(m, percentage)

    @staticmethod
    def setWindowEmissionRatio(material, percentage):
        if not material.node_tree:
            return
        nodes = material.node_tree.nodes
        if "WindowEmissionState" in nodes:
            setCustomNodeValue(
                nodes["WindowEmissionState"],
                "Lit Windows Ratio",
                0.99999 if percentage == 100 else percentage/100.
            )


class MaterialWithColor(MaterialRenderer):
    """
    Material renderer for a general Blender material without UV mapping
    and with some base color
    """
    
    def init(self):
        self.setupMaterial(self.materialName)
    
    def renderForObject(self, obj, slot):
        self.setColorForObject(obj, self.vertexColorLayer, self.b.roofColor)
        slot.material = self.getSingleMaterial()
    
    def onBuildingChanged(self):
        self.updateColorIndex()


class SeamlessTexture(MaterialRenderer):
    
    def init(self):
        self.setupMaterials(self.materialName)
    
    def renderWalls(self, face, width):
        self.setMaterial(face, self.materialName)
    
    def renderRoof(self, face):
        self.setMaterial(face, self.materialName)
    
    def renderForObject(self, obj, slot):
        slot.material = self.getMaterial()

    def onBuildingChanged(self):
        self.updateColorIndex()


class SeamlessTextureWithColor(MaterialRenderer):
    
    def init(self):
        self.requireVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
        
    def renderWalls(self, face, width):
        self.render(face, self.b.wallsColor)
    
    def renderRoof(self, face):
        self.render(face, self.b.roofColor)
    
    def render(self, face, color):
        self.setColor(face, self.vertexColorLayer, color)
        self.setMaterial(face, self.materialName)
    
    def renderForObject(self, obj, slot):
        self.setColorForObject(obj, self.vertexColorLayer, self.b.roofColor)
        slot.material = self.getMaterial()


class SeamlessTextureScaled(MaterialRenderer):
    
    uvLayer = "size"
    
    def init(self):
        self.requireUvLayer(self.uvLayer)
        self.setupMaterials(self.materialName)
    
    def renderForObject(self, obj, slot):
        s = obj.scale
        self.setDataForObject(obj, self.uvLayer, (math.sqrt(s[0]*s[0] + s[1]*s[1]), s[2]))
        slot.material = self.getMaterial()

    def onBuildingChanged(self):
        self.updateColorIndex()


class SeamlessTextureScaledWithColor(MaterialRenderer):
    
    uvLayer = "size"
    
    def init(self):
        self.requireUvLayer(self.uvLayer)
        self.requireVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
    
    def renderForObject(self, obj, slot):
        s = obj.scale
        self.setDataForObject(obj, self.uvLayer, (math.sqrt(s[0]*s[0] + s[1]*s[1]), s[2]))
        self.setColorForObject(obj, self.vertexColorLayer, self.b.roofColor)
        slot.material = self.getMaterial()


class FacadeSeamlessTexture(FacadeWithColor):
    
    uvLayer = "data.1"
        
    def init(self):
        self.requireUvLayer(self.uvLayer)
        if self.colors:
            self.requireVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
        self.setupMaterials(self.materialName2)
    
    def renderWalls(self, face, width):
        # building
        b = self.b
        if self.colors:
            self.setColor(face, self.vertexColorLayer, b.wallsColor)
        if b.z1:
            self.setData(face, self.uvLayer, b.numLevels)
            self.setMaterial(face, self.materialName)
        else:
            self.setData(face, self.uvLayer, b.levelHeights)
            self.setMaterial(face, self.materialName2)


class FacadeWithOverlay(FacadeWithColor):
    
    uvLayer = "data.1"
    
    def __init__(self, renderer, baseMaterialName, *args):
        super().__init__(renderer, baseMaterialName, args[1])
        self.wallMaterial = "%s_color" % args[0]
        self.colorIndex = -1
    
    def init(self):
        self.requireUvLayer(self.uvLayer)
        self.requireVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
        self.setupMaterials(self.materialName2)
        self.setupMaterial(self.wallMaterial)
    
    def renderWalls(self, face, width):
        # building
        b = self.b
        self.setData(face, self.uvLayer, b.levelHeights)
        self.setColor(face, self.vertexColorLayer, b.wallsColor)
        if b.noWindows or width < 1.:
            self.setSingleMaterial(face, self.wallMaterial)
        else:
            if b.z1:
                self.setMaterial(face, self.materialName)
            else:
                self.setMaterial(face, self.materialName2)