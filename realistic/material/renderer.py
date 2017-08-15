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
    
    def setColor(self, face, layerName, color):
        vertexColorLayer = self.r.bm.loops.layers.color[layerName]
        for loop in face.loops:
            loop[vertexColorLayer] = color
    
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
                # <self.groupName> will be used in the code if another group
                self.groupName = groupName
                self.materials = materials
        else:
            print("No materials with the base name %s have been found" % groupName)
        
        self.r.materialGroups.add(groupName)
    
    def setMaterial(self, face, groupName=None):
        r = self.r
        materialIndices = r.materialIndices
        materials = r.obj.data.materials
        if not self.outline is r.outline:
            self.outline = r.outline
            # increase <self.index> to use the next material
            self.index += 1
            if self.index == self.numMaterials:
                # all available materials have been used, so set <self.index> to zero
                self.index = 0
        # the name of the current material
        name = (
            self.materials[groupName if groupName else self.groupName]\
            if self.multipleGroups else\
            self.materials
        )[self.index]
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
    
    def setSingleMaterial(self, face):
        r = self.r
        materialIndices = r.materialIndices
        materials = r.obj.data.materials
        name = self.materialName
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
    
    def getMaterial(self):
        return bpy.data.materials[self.materialName]


class MaterialWithColor(MaterialRenderer):
    """
    Material renderer for a general Blender material without UV mapping
    and with some base color
    """
    
    def init(self):
        self.setupMaterial()
    
    def render(self, face):
        roofColor = self.b.roofColor
        self.setColor(face, self.vertexColorLayer, roofColor if roofColor else self.roofColor)
        self.setSingleMaterial(face)


class SeamlessTexture(MaterialRenderer):
    
    def init(self):
        self.setupMaterials(self.materialName)
    
    def render(self, face):
        self.setMaterial(face, self.materialName)


class SeamlessTextureWithColor(MaterialRenderer):
    
    def init(self):
        self.ensureVertexColorLayer(self.vertexColorLayer)
        self.setupMaterials(self.materialName)
    
    def render(self, face):
        self.setColor(face, self.vertexColorLayer, self.b.roofColor)
        self.setMaterial(face, self.materialName)