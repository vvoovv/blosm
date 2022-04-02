import bpy
from renderer import Renderer
from .item_store import ItemStore
from .asset_store import AssetStore
from .texture_exporter import TextureExporter

from item.building import Building
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

_itemClasses = (
        Building,
        Footprint,
        Facade,
        Level,
        CurtainWall,
        Div,
        Bottom,
        Window,
        Door,
        Balcony,
        Chimney,
        RoofFlat,
        RoofFlatMulti,
        RoofProfile,
        RoofGeneratrix,
        RoofHipped,
        RoofHippedMulti,
        RoofSide
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
        self.facadeRenderer = itemRenderers["Facade"]
        
        self.exportMaterials = app.enableExperimentalFeatures and app.importForExport
        
        if self.exportMaterials:
            self.textureExporter = TextureExporter(self.assetsDir, self.assetPackageDir)
            # Do we need to cache <claddingTextureInfo> for each cladding material?
            self.cacheCladdingTextureInfo = False
        
        # initialize item renderers
        for item in itemRenderers:
            itemRenderers[item].init(itemRenderers, self)
        
        self.getStyle = getStyle
        self.itemStore = ItemStore(_itemClasses)
        
        self.assetStore = AssetStore(app.assetInfoFilepath)
        
        self._cache = {}
        
        self.buildingActions = []
        self.footprintActions = []
        # "rev" stands for "render extruded volumes"
        self.revActions = []
    
    def prepare(self):
        # nothing to be done here for now
        pass

    def preRender(self, building):
        element = building.element
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
        for action in self.buildingActions:
            action.cleanup()
        
        if self.exportMaterials:
            self.textureExporter.cleanup()
        
        self._cache.clear()
    
    def render(self, building, data):
        parts = building.parts
        itemStore = self.itemStore
        
        #if "id" in outline.tags: print(outline.tags["id"]) #DEBUG OSM id
        
        building.renderInfo = Building(data)
        
        # get the style of the building
        buildingStyle = self.styleStore.get(self.getStyle(building, self.app))
        if not buildingStyle:
            # skip the building
            return
        building.renderInfo.setStyleMeta(buildingStyle)
        
        if self.app.renderAfterExtrude:
            self.preRender(building)
        
        if not parts or building.alsoPart:
            # the building has no parts
            footprint = Footprint(building, building)
            # The attribute <footprint> below may be used in calculation of the area of
            # the building footprint or in <action.terrain.Terrain>
            #building.footprint = footprint
            itemStore.add(footprint)
        if parts:
            itemStore.add((Footprint(part, building) for part in parts), Footprint, len(parts))
        
        for actions in (self.buildingActions, self.footprintActions):
            for action in actions:
                action.do(building, buildingStyle, self)
                if itemStore.skip:
                    break
            if itemStore.skip:
                break
        itemStore.clear()
        
        if itemStore.skip:
            itemStore.skip = False
        elif self.app.renderAfterExtrude:
            self.postRender(building.element)
    
    def renderExtrudedVolumes(self, building, data):
        self.preRender(building)
        
        # render building footprint
        if not building.parts or building.alsoPart:
            self.renderExtrudedVolume(building.footprint, True)
        # render building parts
        for part in building.parts:
            self.renderExtrudedVolume(part.footprint, False)
        
        self.postRender(building.element)
    
    def renderExtrudedVolume(self, footprint, isBldgFootprint):
        if not footprint:
            return
        
        for action in self.revActions:
            action.do(footprint)
        
        if not footprint.noWalls:
            self.facadeRenderer.render(footprint)
        footprint.roofRenderer.render(footprint.roofItem)
    
    def createFace(self, building, indices):
        bm = self.bm
        renderInfo = building.renderInfo
        verts = renderInfo.verts
        bmVerts = renderInfo.bmVerts
        
        # extend <bmVerts> to have the same number of vertices as in <verts>
        bmVerts.extend(None for _ in range(len(verts)-len(bmVerts)))
        
        # check if we have BMVerts for for all <indices>
        for index in indices:
            if not bmVerts[index]:
                bmVerts[index] = bm.verts.new(
                    (verts[index] + renderInfo.offset) if renderInfo.offset else verts[index]
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