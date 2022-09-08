from math import floor
from . import ItemRendererTexture
from .corner import Corner
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast
from util import rgbToHex
from util.blender import linkObjectFromFile, createMeshObject
from ..util import getFilepath


def _getTileWidthM(facadeTextureInfo):
    if not "tileWidthM" in facadeTextureInfo:
        # cache <tileWidthM>
        facadeTextureInfo["tileWidthM"] = facadeTextureInfo["textureSize"][0]/\
            (facadeTextureInfo["featureRpx"] - facadeTextureInfo["featureLpx"])*\
            facadeTextureInfo["featureWidthM"] / facadeTextureInfo["numTilesU"]
    return facadeTextureInfo["tileWidthM"]


class RenderState:
    """
    The class instance hold
    """
    
    def __init__(self):
        #
        # Variables for <Container.renderLevels(..)>
        #
        # <indexBL> and <indexBR> are indices of the bottom vertices of an level item to be created
        # in <Container.renderLevels(..)> out of the parent item
        # The prefix <BL> means "bottom left"
        self.indexBL = 0
        # The prefix <BR> means "bottom rights"
        self.indexBR = 0
        # <texVb> is the current V-coordinate for texturing the bottom vertices of
        # level items to be created in <Container.renderLevels(..)> out of the parent item
        self.texVb = 0
        
        #
        # Variables for <Container.renderDivs(..)>
        #
        # <indexLB> and <indexLT> are indices of the leftmost vertices of an item to be created
        # The prefix <LB> means "left bottom"
        self.indexLB = 0
        # The prefix <LT> means "left top"
        self.indexLT = 0
        # <texUl> is the current U-coordinate for texturing the leftmost vertices of
        # items to be created out of <item>
        self.texUl = 0.
        # <texVlt> is the current V-coordinate for texturing the top leftmost vertex of
        # items to be created out of <item>
        self.texVlt = 0.
        # <startIndex> is only used by geometry <TrapezoidChainedRv>.
        # It's auxiliary variable used for optimization purposes. The task is to find intersection of
        # the uppper part of the chained trapezoid with the vertical line with the given X-coordinate.
        # <startIndex> helps to limit the search.
        # One starts with <startIndex> = <n> - 1, where <n> is the number of vertices in
        # the chained trapezoid, i.e. it points to the top leftmost vertex of the chained trapezoid.
        # Then <startIndex> is gradually decreased, i.e. <startIndex> points to a vertex to the left
        # from the previous vertex.
        self.startIndex = 0
        
        self.tmpTriangle = True
        

class Container(ItemRendererTexture):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        self.claddingTexture = True
        self.renderState = RenderState()
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            #face = self.r.createFace(item.footprint, item.indices)
            #self.renderCladding(item, face, item.uvs)
            #return
            if item.highEnoughForLevel:
                self.renderLevels(item)
            else:
                # No space for levels, so we render cladding only.
                self.renderCladding(
                    item,
                    self.r.createFace(item.footprint, item.indices),
                    item.uvs
                )
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        item.levelGroups.init()
        
        for _item in item.markup:
            _item.prepareMarkupItems()
            # inherit the width from <item> to the markup items
            _item.width = item.width
            if _item.markup:
                _item.calculateMarkupDivision(self.r)
        
        item.geometry.renderLevelGroups(item, self)
    
    def renderDivs(self, item, levelGroup):
        # If <levelGroup> is given, that actually means that <item> is a level or contained
        # inside another level item. In this case the call to <self.renderLevelGroup(..)>
        # will be made later in the code
        
        # <r> is the global building renderer
        r = self.r
        parentIndices = item.indices
        geometry = item.geometry
        
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            #item.calculateMarkupDivision(r.assetStore)
            if not item.valid:
                return
            # create vertices for the markup items
            numItems = len(item.markup)
            if numItems == 1:
                # the special case
                _item = item.markup[0]
                _item.indices = parentIndices
                _item.uvs = item.uvs
                r.createFace(item.footprint, _item.indices)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                rs = self.renderState
                geometry.initRenderStateForDivs(rs, item)
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        geometry.renderDivs(
                            self, item, levelGroup,
                            0, numItems, 1,
                            rs
                        )
                        if symmetry:
                            geometry.renderDivs(
                                self, item, levelGroup,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                rs
                            )
                geometry.renderDivs(
                    self, item, levelGroup,
                    0, numItems if symmetry else numItems-1, 1,
                    rs
                )
                if symmetry:
                    geometry.renderDivs(
                        self, item, levelGroup,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        rs
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                if lastItem.width:
                    geometry.renderLastDiv(
                        self, item, levelGroup, lastItem,
                        rs
                    )
        else:
            pass

    def setData(self, face, layerName, uv):
        if not isinstance(uv, tuple):
            uv = (uv, uv)
        uvLayer = self.r.bm.loops.layers.uv[layerName]
        for loop in face.loops:
            loop[uvLayer].uv = uv
    
    def setMaterialId(self, item, facadeTextureInfo, uvs):
        claddingTextureInfo = self.getCladdingTextureInfo(item)\
            if facadeTextureInfo.get("cladding") and self.claddingTexture else\
            None
        
        materialId = self.getFacadeMaterialId(item, facadeTextureInfo, claddingTextureInfo)
        if self.createFacadeMaterial(item, materialId, facadeTextureInfo, claddingTextureInfo, uvs):
            item.materialId = materialId
            item.materialData = facadeTextureInfo, claddingTextureInfo
        else:
            item.materialId = ""
    
    def renderLevelGroup(self, item, levelGroup, indices, uvs):
        """
        Args:
            item (item.container.Container): It's needed to get <building> and
                width for the placeholder of <levelGroup>. Example configurations:
                facade.level (<referenceItem> is facade)
                facade.div.level (<referenceItem> is div)
                facade.level.div (<referenceItem> is div)
                facade.div.level.div (<referenceItem> is the second div)
            levelGroup: level group
            indices (list or tuple): indices of vertices
            uvs (list or tuple): UV-coordinates of the vertices
        """
        assetInfo = None
        
        if levelGroup.item:
            # asset info could have been set in the call to item.getWidth(..)
            assetInfo = item.assetInfo
            # if <assetInfo=0>, then it was already queried in the asset store and nothing was found
            if assetInfo is None:
                assetInfo = self.getAssetInfo(item)
            
            if assetInfo:
                if assetInfo["type"] == "mesh":
                    #
                    # mesh
                    #
                    
                    if "object" in assetInfo:
                        #
                        # a separate Blender object
                        #
                        
                        # first we check for corner modules (mesh only) defined implicitly
                        if floor(item.width/assetInfo["tileWidthM"]) > 1:
                            if item.cornerL or item.cornerR:
                                assetInfoCorner = self.getAssetInfoCorner(item, assetInfo["class"])
                                if assetInfoCorner and "collection" in assetInfoCorner:
                                    self.processImplicitCornerItems(item, levelGroup, indices, assetInfo, assetInfoCorner)
                                    return
                        
                        objName = assetInfo["object"]
                        
                        # If <objectName> isn't available in <meshAssets>, that also means
                        # that <objectName> isn't available in <self.r.buildingAssetsCollection.objects>
                        if not objName in self.r.meshAssets:
                            obj = linkObjectFromFile(getFilepath(self.r, assetInfo), None, objName)
                            if not obj:
                                return
                            self.processAssetMeshObject(obj, objName)
                    else:
                        #
                        # a Blender collection (e.g. of corner modules)
                        #
                        obj = self.processCollection(item, assetInfo, indices)
                        if not obj:
                            # something is wrong or no actions are needed
                            return
                        objName = obj.name
                        if not objName in self.r.meshAssets:
                            self.processAssetMeshObject(obj, objName)
                    
                    self.prepareGnVerts(
                        item, levelGroup, indices, assetInfo,
                        self.getGnInstanceObject(item, objName)
                    )
                else:
                    #
                    # texture
                    #
                    face = self.r.createFace(item.footprint, indices)
                    
                    if item.materialId is None:
                        self.setMaterialId(
                            item,
                            assetInfo,
                            uvs
                        )
                    if item.materialId:
                        facadeTextureInfo, claddingTextureInfo = item.materialData
                        layer = item.building.element.l
                        self.r.setUvs(
                            face,
                            self.getUvs(item, levelGroup, facadeTextureInfo),
                            layer,
                            layer.uvLayerNameFacade
                        )
                        self.renderExtra(item, face, facadeTextureInfo, claddingTextureInfo, uvs)
                        self.r.setMaterial(layer, face, item.materialId)
                    else:
                        self.renderCladding(item, face, uvs)
        
        if not assetInfo:
            self.renderCladding(
                item,
                self.r.createFace(item.footprint, indices),
                uvs
            )
            item.materialId = ""
    
    def getGnInstanceObject(self, item, objName):
        obj, params, instances = self.r.meshAssets[objName]
        
        if params:
            # create a key out of <properties>
            key = '_'.join(
                rgbToHex( item.getStyleBlockAttrDeep(_param) or obj[_param] ) for _param in params
            )
            if not key in instances:
                # the created Blender object <_obj> that shares the mesh data with <obj>
                _obj = instances[key] = createMeshObject(objName, mesh = obj.data, collection = self.r.buildingAssetsCollection)
                # set properties
                for _param in params:
                    _obj[_param] = item.getStyleBlockAttrDeep(_param) or obj[_param]
            obj = instances[key]
        
        return obj
    
    def getUvs(self, item, levelGroup, facadeTextureInfo):
        return item.geometry.getFinalUvs(
            max( round(item.width/_getTileWidthM(facadeTextureInfo)), 1 ),
            self.getNumLevelsInFace(levelGroup),
            facadeTextureInfo["numTilesU"],
            facadeTextureInfo["numTilesV"]
        )
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)
    
    def processAssetMeshObject(self, obj, objName):
        # We need only the properties of a Blender object created by a designer.
        # <rna_properties> is used to filter our the other properties
        rna_properties = {
            prop.identifier for prop in obj.bl_rna.properties if prop.is_runtime
        }
        params = [_property[2:] for _property in obj.keys() if not _property in rna_properties and _property.startswith("p_")]
        self.r.meshAssets[objName] = (obj, params, {} if params else None)
        
        if not params:
            # use <obj> as is (i.e. linked from another Blender file)
            self.r.buildingAssetsCollection.objects.link(obj)
    
    def prepareGnVerts(self, item, levelGroup, indices, assetInfo, obj):
        layer = item.building.element.l
        
        tileWidth = assetInfo["tileWidthM"]
        
        numTilesX = max( floor(item.width/tileWidth), 1 )
        numTilesY = 1 if levelGroup.singleLevel else levelGroup.index2 - levelGroup.index1 + 1
        scaleX = item.width/(numTilesX*tileWidth)
        scaleY = levelGroup.levelHeight/assetInfo["tileHeightM"]
        
        tileWidth *= scaleX
        
        # increment along X-axis of <item>
        incrementVector = tileWidth * item.facade.vector.unitVector3d
        
        bmVerts = layer.bmGn.verts
        attributeValuesGn = layer.attributeValuesGn
        
        _vertLocation = item.building.renderInfo.verts[indices[0]] + 0.5*incrementVector
        if numTilesY == 1:
            for _ in range(numTilesX):
                bmVerts.new(_vertLocation)
                _vertLocation += incrementVector
                attributeValuesGn.append((
                    obj.name,
                    item.facade.vector.vector3d,
                    scaleX,
                    scaleY
                ))
        else:
            for _ in range(numTilesY):
                vertLocation = _vertLocation.copy()
                for _ in range(numTilesX):
                    bmVerts.new(vertLocation)
                    attributeValuesGn.append((
                        obj.name,
                        item.facade.vector.vector3d,
                        scaleX,
                        scaleY
                    ))
                    vertLocation += incrementVector
                _vertLocation[2] += levelGroup.levelHeight
    
    def processImplicitCornerItems(self, item, levelGroup, indices, assetInfo, assetInfoCorner):
        # the same code as in self.prepareGnVerts(..)
        # the beginning of the code >
        layer = item.building.element.l
        
        tileWidth = assetInfo["tileWidthM"]
        
        numTilesX = max( floor(item.width/tileWidth), 1 )
        numTilesY = 1 if levelGroup.singleLevel else levelGroup.index2 - levelGroup.index1 + 1
        scaleX = item.width/(numTilesX*tileWidth)
        scaleY = levelGroup.levelHeight/assetInfo["tileHeightM"]
        
        tileWidth *= scaleX
        
        # increment along X-axis of <item>
        incrementVector = tileWidth * item.facade.vector.unitVector3d
        
        bmVerts = layer.bmGn.verts
        attributeValuesGn = layer.attributeValuesGn
        # the end of the code <
        
        if item.cornerL:
            self._processImplicitCornerItem(
                item, levelGroup, assetInfoCorner, True,
                item.building.renderInfo.verts[indices[0]] + incrementVector
            )
        if item.cornerR:
            self._processImplicitCornerItem(
                item, levelGroup, assetInfoCorner, False,
                item.building.renderInfo.verts[indices[1]] - incrementVector
            )
        
        objName = assetInfo["object"]
        
        # If <objectName> isn't available in <meshAssets>, that also means
        # that <objectName> isn't available in <self.r.buildingAssetsCollection.objects>
        if not objName in self.r.meshAssets:
            obj = linkObjectFromFile(getFilepath(self.r, assetInfo), None, objName)
            if not obj:
                return
            self.processAssetMeshObject(obj, objName)
        
        obj = self.getGnInstanceObject(item, objName)
        
        _vertLocation = item.building.renderInfo.verts[indices[0]] + (1.5 if item.cornerL else 0.5) *incrementVector
        if numTilesY == 1:
            for _ in range(1 if item.cornerL else 0, numTilesX-1 if item.cornerR else numTilesX):
                # the same code as in self.prepareGnVerts(..)
                bmVerts.new(_vertLocation)
                _vertLocation += incrementVector
                attributeValuesGn.append((
                    obj.name,
                    item.facade.vector.vector3d,
                    scaleX,
                    scaleY
                ))
        else:
            for _ in range(numTilesY):
                vertLocation = _vertLocation.copy()
                for _ in range(1 if item.cornerL else 0, numTilesX-1 if item.cornerR else numTilesX):
                    # the same code as in self.prepareGnVerts(..)
                    bmVerts.new(vertLocation)
                    attributeValuesGn.append((
                        obj.name,
                        item.facade.vector.vector3d,
                        scaleX,
                        scaleY
                    ))
                    vertLocation += incrementVector
                _vertLocation[2] += levelGroup.levelHeight
    
    def getAssetInfoCorner(self, item, baseClass):
        building, collection =\
            item.building, item.getStyleBlockAttrDeep("collection")
        
        return self.app.assetStore.getAssetInfo(
            True,
            building,
            collection,
            "corner",
            baseClass + "_corner"
        )
    
    def _processImplicitCornerItem(self, item, levelGroup, assetInfoCorner, cornerL, cornerVert):
        obj = Corner.processCorner(self,
            item, assetInfoCorner, cornerL,
            cornerVert
        )
        if obj:
            objName = obj.name
            if not objName in self.r.meshAssets:
                self.processAssetMeshObject(obj, objName)
            
            Corner.prepareGnVerts(self,
                item, levelGroup, None, assetInfoCorner,
                self.getGnInstanceObject(item, objName)
            )