import os
import bpy
from mathutils import Vector
from building.renderer import Renderer
from .item_store import ItemStore
from .item_factory import ItemFactory
from .asset_store import AssetStore
from .texture_exporter import TextureExporter
from util.polygon import Polygon

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
        # <self.outlinePolygon> is used only in the case if the buildings has parts
        self.outlinePolygon = Polygon()
    
    def init(self, outline):
        self.verts.clear()
        self.bmVerts.clear()
        # <outline> is an instance of the class as defined by the data model (e.g. parse.osm.way.Way) 
        self.outline = outline
        if self.outlinePolygon.allVerts:
            self.outlinePolygon.clear()
        self.offset = None
        # Instance of item.footprint.Footprint, it's only used if the building definition
        # in the data model doesn't contain building parts, i.e. the building is defined completely
        # by its outline
        self.footprint = None
        self._cache.clear()
        self.metaStyleBlock = None
        self.assetInfoBldgIndex = None
        self._area = 0.
    
    def clone(self):
        building = Building()
        return building

    def attr(self, attr):
        return self.outline.tags.get(attr)

    def __getitem__(self, attr):
        """
        That variant of <self.attr(..) is used in a setup script>
        """
        return self.outline.tags.get(attr)
    
    @classmethod
    def getItem(cls, itemFactory, outline, data):
        item = itemFactory.getItem(cls)
        item.init(outline)
        item.data = data
        return item
    
    def setStyleMeta(self, style):
        if style.meta:
            self.metaStyleBlock = style.meta
        self.use = style.meta.attrs.get("buildingUse") if style.meta else None
    
    def area(self):
        if not self._area:
            # remember that <self.footprint> is defined if the building doesn't have parts
            polygon = self.footprint.polygon if self.footprint else self.outlinePolygon
            
            if not polygon.allVerts:
                outline = self.outline
                if outline.t is Renderer.multipolygon:
                    coords = outline.getOuterData(self.data)
                else:
                    coords = outline.getData(self.data)
                polygon.init( Vector(coord) for coord in coords )
            if polygon.n < 3:
                # the building will be skipped later in method <calculated>
                return 0.
            
            self._area = polygon.area()
        return self._area


class BuildingRendererNew(Renderer):
    
    assetInfoFilename = "default.json"
    
    def __init__(self, app, styleStore, itemRenderers, getStyle=None):
        self.app = app
        app.addRenderer(self)
        
        # offset for a Blender object created if <layer.singleObject is False>
        self.offset = None
        
        self.styleStore = styleStore
        
        self.assetsDir = app.assetsDir
        
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
            self.textureExporter = TextureExporter(self.assetsDir)
            # Do we need to cache <claddingTextureInfo> for each cladding material?
            self.cacheCladdingTextureInfo = False
        
        # initialize item renderers
        for item in itemRenderers:
            itemRenderers[item].init(itemRenderers, self)
        
        self.getStyle = getStyle
        referenceItems = _createReferenceItems(app)
        self.itemStore = ItemStore(referenceItems)
        self.itemFactory = ItemFactory(referenceItems)
        
        assetInfoFilepath = app.assetInfoFilepath
        if not assetInfoFilepath:
            assetInfoFilepath = "%s_256.json" % self.assetInfoFilename[:-5] if exportMaterials else self.assetInfoFilename
            assetInfoFilepath = os.path.join(
                os.path.dirname(os.path.abspath(app.bldgMaterialsFilepath)),
                "asset_info",
                assetInfoFilepath
            )
        self.assetStore = AssetStore(assetInfoFilepath)
        
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
        if self.exportMaterials:
            self.textureExporter.cleanup()
        
        self._cache.clear()
    
    def render(self, buildingP, data):
        parts = buildingP.parts
        itemFactory = self.itemFactory
        itemStore = self.itemStore
        
        # <buildingP> means "building from the parser"
        outline = buildingP.outline
        
        #if "id" in outline.tags: print(outline.tags["id"]) #DEBUG OSM id
        
        building = Building.getItem(itemFactory, outline, data)
        
        self.preRender(building)
        
        partTag = outline.tags.get("building:part")
        if not parts or (partTag and partTag != "no"):
            # the building has no parts
            footprint = Footprint.getItem(itemFactory, outline, building)
            # The attribute <footprint> below may be used in calculation of the area of
            # the building footprint or in <action.terrain.Terrain>
            building.footprint = footprint
            itemStore.add(footprint)
        if parts:
            itemStore.add((Footprint.getItem(itemFactory, part, building) for part in parts), Footprint, len(parts))

        # get the style of the building
        buildingStyle = self.styleStore.get(self.getStyle(building, self.app))
        building.setStyleMeta(buildingStyle)
        
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