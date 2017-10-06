import math
import bpy
from util.blender import appendMaterialsFromFile


class MaterialRenderer:
    
    vertexColorLayer = "Col"
    
    # default roof color used by some material renderers
    roofColor = (0.29, 0.25, 0.21)
    
    def __init__(self, renderer, baseMaterialName):
        self.r = renderer
        # base name for Blender materials
        self.materialName = baseMaterialName
        # index of the Blender material to set to a BMFace
        self.index = -1
        # keep track of building outline
        self.outline = None
        # a data structure to store names of Blender materials
        self.materials = None
        # Do we have multiple groups of materials
        # (e.g. apartments and apartments_with_ground_level)?
        self.multipleGroups = False
    
    def ensureUvLayer(self, name):
        uv = self.r.bm.loops.layers.uv
        # create a data UV layer
        if not name in uv:
            uv.new(name)
    
    def ensureVertexColorLayer(self, name):
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
        vertexColorLayer = self.r.bm.loops.layers.color[layerName]
        for loop in face.loops:
            loop[vertexColorLayer] = color
    
    def setColorForObject(self, obj, layerName, color):
        vertexColorLayer = obj.data.vertex_colors[layerName]
        for d in vertexColorLayer.data:
            d.color = color
    
    def setupMaterial(self):
        """
        Unlike <self.setupMaterials(..)> the method setups a single Blender material
        with the name <self.materialName>
        """
        name = self.materialName
        if not name in bpy.data.materials:
            if not appendMaterialsFromFile(self.r.app.bldgMaterialsFilepath, name):
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
            loadedMaterials = appendMaterialsFromFile(
                self.r.app.bldgMaterialsFilepath,
                *materialsToLoad
            )
            materials.extend(m.name for m in loadedMaterials if not m is None)
        
        if materials:
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
            print("No materials with the base name %s have been found" % groupName)
        
        self.r.materialGroups.add(groupName)
    
    def setMaterial(self, face, groupName=None):
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
        r = self.r
        if not self.outline is r.outline:
            self.outline = r.outline
            # increase <self.index> to use the next material
            self.index += 1
            if self.index == self.numMaterials:
                # all available materials have been used, so set <self.index> to zero
                self.index = 0
        # the name of the current material
        return (
            self.materials[groupName if groupName else self.groupName]\
            if self.multipleGroups else\
            self.materials
        )[self.index]
    
    def setSingleMaterial(self, face):
        r = self.r
        materialIndices = r.materialIndices
        materials = r.obj.data.materials
        name = self.materialName
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
    
    def getSingleMaterial(self):
        return bpy.data.materials[self.materialName]


class MaterialWithColor(MaterialRenderer):
    """
    Material renderer for a general Blender material without UV mapping
    and with some base color
    """
    
    def init(self):
        self.setupMaterial()
    
    def renderForObject(self, obj, slot):
        self.setColorForObject(obj, self.vertexColorLayer, self.b.roofColor)
        slot.material = self.getSingleMaterial()


class SeamlessTexture(MaterialRenderer):
    
    def init(self):
        self.setupMaterials(self.materialName)
    
    def renderWalls(self, face):
        self.setMaterial(face, self.materialName)
    
    def renderRoof(self, face):
        self.setMaterial(face, self.materialName)
    
    def renderForObject(self, obj, slot):
        slot.material = self.getMaterial()


class SeamlessTextureWithColor(MaterialRenderer):
    
    def init(self):
        self.ensureVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
        
    def renderWalls(self, face):
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
        self.ensureUvLayer(self.uvLayer)
        self.setupMaterials(self.materialName)
    
    def renderForObject(self, obj, slot):
        s = obj.scale
        self.setDataForObject(obj, self.uvLayer, (math.sqrt(s[0]*s[0] + s[1]*s[1]), s[2]))
        slot.material = self.getMaterial()


class SeamlessTextureScaledWithColor(MaterialRenderer):
    
    uvLayer = "size"
    
    def init(self):
        self.ensureUvLayer(self.uvLayer)
        self.ensureVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
    
    def renderForObject(self, obj, slot):
        s = obj.scale
        self.setDataForObject(obj, self.uvLayer, (math.sqrt(s[0]*s[0] + s[1]*s[1]), s[2]))
        self.setColorForObject(obj, self.vertexColorLayer, self.b.roofColor)
        slot.material = self.getMaterial()


class FacadeSeamlessTexture(MaterialRenderer):
    
    uvLayer = "data.1"
    
    def __init__(self, renderer, baseMaterialName):
        super().__init__(renderer, baseMaterialName)
        self.materialName2 = "%s_with_ground_level" % baseMaterialName
        
    def init(self):
        self.ensureUvLayer(self.uvLayer)
        self.setupMaterials(self.materialName)
        self.setupMaterials(self.materialName2)
        
    def renderWalls(self, face):
        # building
        b = self.b
        if b.z1:
            self.setData(face, self.uvLayer, b.numLevels)
            self.setMaterial(face, self.materialName)
        else:
            self.setData(face, self.uvLayer, b.levelHeights)
            self.setMaterial(face, self.materialName2)