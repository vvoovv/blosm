import bpy
from . import Building
from renderer import Renderer
from .item_store import ItemStore
from .item_factory import ItemFactory
from .asset_store import AssetStore
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
from item.roof_hipped import RoofHipped
from item.roof_hipped_multi import RoofHippedMulti
from item.roof_side import RoofSide


def _createReferenceItems(app):
    return (
        (Building(), 5),
        Footprint(app.buildingEntranceAttr),
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
        RoofHipped(),
        RoofHippedMulti(),
        RoofSide()
    )


class BuildingRendererNew(Renderer):
    
    def __init__(self, app, styleStore, itemRenderers, getStyle=None):
        self.app = app
        app.addRenderer(self)
        
        # offset for a Blender object created if <layer.singleObject is False>
        self.offset = None
        
        self.styleStore = styleStore
        
        self.assetsDir = app.assetsDir
        self.assetPackageDir = app.assetPackageDir
        
        # do wee need to apply a cladding color for facade textures?
        self.useCladdingColor = True
        
        self.itemRenderers = itemRenderers
        
        self.exportMaterials = app.enableExperimentalFeatures and app.importForExport
        
        if self.exportMaterials:
            self.textureExporter = TextureExporter(self.assetsDir, self.assetPackageDir)
            # Do we need to cache <claddingTextureInfo> for each cladding material?
            self.cacheCladdingTextureInfo = False
        
        # initialize item renderers
        for item in itemRenderers:
            itemRenderers[item].init(itemRenderers, self)
        
        self.getStyle = getStyle
        referenceItems = _createReferenceItems(app)
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
        
        self.assetStore = AssetStore(app.assetInfoFilepath)
        
        self._cache = {}
    
    def prepare(self):
        # nothing to be done here for now
        pass

    def preRender(self, building):
        element = building.outline
        layer = element.l
        self.layer = element.l
        
        if layer.singleObject:
            if not layer.bm:
                layer.obj = self.createBlenderObject(
                    layer.name,
                    layer.location,
                    collection = self.collection,
                    parent = None
                )
                layer.prepare(layer)
            self.bm = layer.bm
            self.obj = layer.obj
            self.materialIndices = layer.materialIndices
    
    def cleanup(self):
        if Building.actions:
            for action in Building.actions:
                action.cleanup()
        
        if self.exportMaterials:
            self.textureExporter.cleanup()
        
        self._cache.clear()
    
    def render(self, building, data):
        parts = building.parts
        itemFactory = self.itemFactory
        itemStore = self.itemStore
        
        # <buildingP> means "building from the parser"
        
        #if "id" in outline.tags: print(outline.tags["id"]) #DEBUG OSM id
        
        building.renderInfo = Building.getItem(itemFactory, data)
        
        # get the style of the building
        buildingStyle = self.styleStore.get(self.getStyle(building, self.app))
        if not buildingStyle:
            # skip the building
            return
        building.renderInfo.setStyleMeta(buildingStyle)
        
        self.preRender(building)
        
        partTag = building.outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = Footprint.getItem(itemFactory, building, building)
            # The attribute <footprint> below may be used in calculation of the area of
            # the building footprint or in <action.terrain.Terrain>
            #building.footprint = footprint
            itemStore.add(footprint)
        if parts:
            itemStore.add((Footprint.getItem(itemFactory, part, building) for part in parts), Footprint, len(parts))
        
        for itemClass in (Building, Footprint):
            for action in itemClass.actions:
                action.do(building, itemClass, buildingStyle, self)
                if itemStore.skip:
                    break
            if itemStore.skip:
                break
        itemStore.clear()
        
        if itemStore.skip:
            itemStore.skip = False
        else:
            self.postRender(building.outline)
    
    def createFace(self, building, indices):
        bm = self.bm
        verts = building.verts
        bmVerts = building.bmVerts
        
        # extend <bmVerts> to have the same number of vertices as in <verts>
        bmVerts.extend(None for _ in range(len(verts)-len(bmVerts)))
        
        # check if we have BMVerts for for all <indices>
        for index in indices:
            if not bmVerts[index]:
                bmVerts[index] = bm.verts.new(
                    (verts[index] + building.offset) if building.offset else verts[index]
                )
        
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