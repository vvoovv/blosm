import os
import bpy
from building.renderer import Renderer
from .item_store import ItemStore
from .item_factory import ItemFactory
from .texture_store_facade import FacadeTextureStore
from .texture_store_cladding import CladdingTextureStore
from .texture_exporter import TextureExporter

from item.footprint import Footprint
from item.facade import Facade
from item.level import Level, CurtainWall
from item.div import Div
from item.bottom import Bottom
from item.window import Window
from item.door import Door
from item.balcony import Balcony
from item.chimney import Chimney

from item.roof_flat import RoofFlat
from item.roof_flat_multi import RoofFlatMulti
from item.roof_profile import RoofProfile
from item.roof_generatrix import RoofGeneratrix
from item.roof_side import RoofSide


def _createReferenceItems():
    return (
        (Building(), 5),
        Footprint(),
        Facade(),
        Level(),
        CurtainWall(),
        Div(),
        Bottom(),
        Window(),
        Door(),
        Balcony(),
        Chimney(),
        RoofFlat(),
        RoofFlatMulti(),
        RoofProfile(),
        RoofGeneratrix(),
        RoofSide()
    )


class Building:
    """
    A class representing the building for the renderer
    """
    
    def __init__(self):
        self.verts = []
        # counterparts for <self.verts> in the BMesh
        self.bmVerts = []
        # A cache to store different stuff:
        # attributes evaluated per building rather than per footprint, cladding texture info
        self._cache = {}
    
    def init(self, outline):
        self.verts.clear()
        self.bmVerts.clear()
        # <outline> is an instance of the class as defined by the data model (e.g. parse.osm.way.Way) 
        self.outline = outline
        self.offsetZ = None
        # Instance of item.footprint.Footprint, it's only used if the building definition
        # in the data model doesn't contain building parts, i.e. the building is defined completely
        # by its outline
        self.footprint = None
        self._cache.clear()
        self.metaStyleBlock = None
    
    def clone(self):
        building = Building()
        return building
    
    @classmethod
    def getItem(cls, itemFactory, outline, style):
        item = itemFactory.getItem(cls)
        item.init(outline)
        if style.meta:
            item.metaStyleBlock = style.meta
        return item


class BuildingRendererNew(Renderer):
    
    def __init__(self, app, styleStore, itemRenderers, getStyle=None):
        self.app = app
        app.addRenderer(self)
        
        # offset for a Blender object created if <layer.singleObject is False>
        self.offset = None
        # offset if a terrain is set (used instead of <self.offset>)
        self.offsetZ = None
        
        self.styleStore = styleStore
        
        self.bldgMaterialsDirectory = os.path.dirname(app.bldgMaterialsFilepath)
        
        # do wee need to apply a cladding color for facade textures?
        self.useCladdingColor = True
        
        # do we need export materials?
        exportMaterials = False
        
        self.itemRenderers = itemRenderers
        # check if need to export materials
        for item in itemRenderers:
            # If at least one item renderer creates materials for export,
            # then we set <self.exportMaterial> (i.e. for the global renderer) to <True>
            if not exportMaterials and itemRenderers[item].exportMaterials:
                exportMaterials = True
        
        self.exportMaterials = exportMaterials
        if exportMaterials:
            self.textureExporter = TextureExporter(self.bldgMaterialsDirectory)
            # Do we need to cache <claddingTextureInfo> for each cladding material?
            self.cacheCladdingTextureInfo = True
        
        # initialize item renderers
        for item in itemRenderers:
            itemRenderers[item].init(itemRenderers, self)
        
        self.getStyle = getStyle
        referenceItems = _createReferenceItems()
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
        
        self.facadeTextureStore = FacadeTextureStore()
        self.claddingTextureStore = CladdingTextureStore()
        
        self._cache = {}
    
    def prepare(self):
        # nothing to be done here for now
        pass
    
    def cleanup(self):
        if self.exportMaterials:
            self.textureExporter.cleanup()
        
        self._cache.clear()
    
    def render(self, buildingP, data):
        parts = buildingP.parts
        itemFactory = self.itemFactory
        itemStore = self.itemStore
        
        # get the style of the building
        buildingStyle = self.styleStore.get(self.getStyle(buildingP, self.app))
        
        # <buildingP> means "building from the parser"
        outline = buildingP.outline
        
        self.preRender(outline)
        #for itemRenderer in self.itemRenderers:
        #    self.itemRenderers[itemRenderer].preRender()
        
        building = Building.getItem(itemFactory, outline, buildingStyle)
        partTag = outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = Footprint.getItem(itemFactory, outline, building)
            # this attribute <footprint> below may be used in <action.terrain.Terrain>
            building.footprint = footprint
            itemStore.add(footprint)
        if parts:
            itemStore.add((Footprint.getItem(itemFactory, part, building) for part in parts), Footprint, len(parts))
        
        for itemClass in (Building, Footprint):
            for action in itemClass.actions:
                action.do(building, itemClass, buildingStyle)
                if itemStore.skip:
                    break
            if itemStore.skip:
                itemStore.skip = False
                break
        itemStore.clear()
        
        self.postRender(outline)
    
    def createFace(self, building, indices):
        bm = self.bm
        verts = building.verts
        bmVerts = building.bmVerts
        
        # extend <bmVerts> to have the same number of vertices as in <verts>
        bmVerts.extend(None for _ in range(len(verts)-len(bmVerts)))
        
        # check if we have BMVerts for for all <indices>
        for index in indices:
            if not bmVerts[index]:
                bmVerts[index] = bm.verts.new( verts[index] )
        
        return bm.faces.new(bmVerts[index] for index in indices)
    
    def setUvs(self, face, uvs, layerName):
        # assign uv coordinates
        uvLayer = self.bm.loops.layers.uv[layerName]
        loops = face.loops
        for loop,uv in zip(loops, uvs):
            loop[uvLayer].uv = uv
    
    def setVertexColor(self, face, color, layerName):
        vertexColorLayer = self.bm.loops.layers.color[layerName]
        for loop in face.loops:
            loop[vertexColorLayer] = color

    def setMaterial(self, face, materialName):
        """
        Set material (actually material index) for the given <face>.
        """
        materialIndices = self.materialIndices
        materials = self.obj.data.materials
        
        if not materialName in materialIndices:
            materialIndices[materialName] = len(materials)
            materials.append(bpy.data.materials[materialName] if materialName else None)
        face.material_index = materialIndices[materialName]