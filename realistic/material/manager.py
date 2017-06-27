import bpy
from util.blender import appendMaterialsFromFile


class MaterialManager:
    
    def __init__(self, renderer):
        self.r = renderer
        # index of the Blender material to set to a BMFace
        self.index = -1
        # keep track of building outline
        self.outline = None
    
    def ensureUvLayer(self, name):
        uv = self.r.bm.loops.layers.uv
        # create a data UV layer
        if not name in uv:
            uv.new(name)
    
    def setData(self, face, uvLayerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, 0.)
        uvLayer = self.r.bm.loops.layers.uv[uvLayerName]
        for loop in face.loops:
            loop[uvLayer].uv = uv
    
    @property
    def numLevels(self):
        return self.b.getNumLevels()
    
    @property
    def levelHeights(self):
        return self.b.getLevelHeights()
    
    def setupMaterials(self, baseName, numMaterials=20):
        # names of materials to load from a .blen file
        materialsToLoad = []
        # names of materials available after scanning everything
        materials = []
        
        for name in ("%s.%s" % (baseName, i) for i in range(1, numMaterials+1)):
            if name in bpy.data.materials:
                materials.append(name)
            else:
                materialsToLoad.append(name)
        if materialsToLoad:
            loadedMaterials = appendMaterialsFromFile(
                "D:\\projects\\prokitektura\\projects\\building material\\building_material_background.blend",
                *materialsToLoad
            )
            materials.extend(m.name for m in loadedMaterials if not m is None)
        self.materials = materials
        self.numMaterials = len(materials)
    
    def setMaterial(self, face):
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
        name = self.materials[self.index]
        if not name in materialIndices:
            materialIndices[name] = len(materials)
            materials.append(bpy.data.materials[name])
        face.material_index = materialIndices[name]
        