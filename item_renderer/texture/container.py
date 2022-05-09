from . import ItemRendererTexture
from .. import _setAssetInfoCache
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast


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
            #face = self.r.createFace(item.building, item.indices)
            #self.renderCladding(item, face, item.uvs)
            #return
            if item.highEnoughForLevel:
                self.renderLevels(item)
            else:
                # No space for levels, so we render cladding only.
                self.renderCladding(
                    item,
                    self.r.createFace(item.building, item.indices),
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
        
        # calculate number of repeats in the method below
        item.finalizeMarkupDivision()
        
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
                r.createFace(item.building, _item.indices)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                rs = self.renderState
                geometry.initRenderStateForDivs(rs, item)
                # a unit vector along U-axis (the horizontal axis)
                unitVector = item.parent.vector.unitVector
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        geometry.renderDivs(
                            self, item, levelGroup, unitVector,
                            0, numItems, 1,
                            rs
                        )
                        if symmetry:
                            geometry.renderDivs(
                                self, item, levelGroup, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                rs
                            )
                geometry.renderDivs(
                    self, item, levelGroup, unitVector,
                    0, numItems if symmetry else numItems-1, 1,
                    rs
                )
                if symmetry:
                    geometry.renderDivs(
                        self, item, levelGroup, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        rs
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
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
    
    def setMaterialId(self, item, buildingPart, uvs):
        building = item.building
        renderInfo = building.renderInfo
        
        # asset info could have been set in the call to item.getWidth(..)
        facadeTextureInfo = item.assetInfo
        if not facadeTextureInfo:
            # get a texture that fits to the Level markup pattern
            if renderInfo.assetInfoBldgIndex is None:
                facadeTextureInfo = self.r.assetStore.getAssetInfo(building, buildingPart, "texture")
                _setAssetInfoCache(
                    building,
                    facadeTextureInfo,
                    # here <p> is for part
                    "p%s" % buildingPart
                )
            else:
                key = "p%s" % buildingPart
                # If <key> is available in <renderInfo._cache>, that means we'll get <assetInfo> for sure
                facadeTextureInfo = self.r.assetStore.getAssetInfoByBldgIndex(
                    renderInfo._cache[key] if key in renderInfo._cache else renderInfo.assetInfoBldgIndex,
                    buildingPart,
                    "texture"
                )
                if not facadeTextureInfo:
                    # <key> isn't available in <renderInfo._cache>, so <renderInfo.assetInfoBldgIndex> was used
                    # in the call above. No we try to get <facadeTextureInfo> without <renderInfo.assetInfoBldgIndex>
                    facadeTextureInfo = self.r.assetStore.getAssetInfo(building, buildingPart, "texture")
                    _setAssetInfoCache(building, facadeTextureInfo, key)
        
        if facadeTextureInfo:
            claddingTextureInfo = self.getCladdingTextureInfo(item)\
                if facadeTextureInfo.get("cladding") and self.claddingTexture else\
                None
            
            materialId = self.getFacadeMaterialId(item, facadeTextureInfo, claddingTextureInfo)
            if self.createFacadeMaterial(item, materialId, facadeTextureInfo, claddingTextureInfo, uvs):
                item.materialId = materialId
                item.materialData = facadeTextureInfo, claddingTextureInfo
            else:
                item.materialId = ""
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
        building = item.building      
        face = self.r.createFace(building, indices)
        if levelGroup.item:
            if item.materialId is None:
                self.setMaterialId(
                    item,
                    levelGroup.buildingPart or levelGroup.item.buildingPart,
                    uvs
                )
            if item.materialId:
                facadeTextureInfo, claddingTextureInfo = item.materialData
                self.r.setUvs(
                    face,
                    item.geometry.getFinalUvs(
                        max( round(item.width/_getTileWidthM(facadeTextureInfo)), 1 ),
                        self.getNumLevelsInFace(levelGroup),
                        facadeTextureInfo["numTilesU"],
                        facadeTextureInfo["numTilesV"]
                    ),
                    self.r.layer.uvLayerNameFacade
                )
                self.renderExtra(item, face, facadeTextureInfo, claddingTextureInfo, uvs)
                self.r.setMaterial(face, item.materialId)
            else:
                self.renderCladding(item, face, uvs)
        else:
            self.renderCladding(item, face, uvs)
    
    def getNumLevelsInFace(self, levelGroup):
        return 1 if levelGroup.singleLevel else (levelGroup.index2-levelGroup.index1+1)