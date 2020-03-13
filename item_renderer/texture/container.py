from .. import ItemRenderer
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast


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
        

class Container(ItemRenderer):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        self.renderState = RenderState()

    def getMarkupItemRenderer(self, item):
        """
        Get a renderer for the <item> contained in the markup.
        """
        return self.itemRenderers[item.__class__.__name__]

    def getLevelRenderer(self, levelGroup):
        """
        Get a renderer for the <levelGroup> representing <levelGroup.item>.
        <levelGroup.item> is contained in the markup.
        """
        item = levelGroup.item
        # here is the special case: the door which the only item in the markup
        return self.itemRenderers["Door"]\
            if len(item.markup) == 1 and item.markup[0].__class__.__name__ == "Door"\
            else self.itemRenderers["Level"]
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            #face = self.r.createFace(item.building, item.indices)
            #self.renderCladding(item.building, item, face, item.uvs)
            #return
            self.renderLevels(item)
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        geometry = item.geometry
        levelGroups = item.levelGroups
        levelGroups.init()
        # sanity check
        width = item.getWidthForVerticalArrangement()
        if width > item.width:
            item.valid = False
            return
        
        geometry.renderLevelGroups(item, self)
    
    def renderDivs(self, item):
        # <r> is the global building renderer
        r = self.r
        building = item.building
        parentIndices = item.indices
        geometry = item.geometry
        
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            item.calculateMarkupDivision()
            if not item.valid:
                return
            # create vertices for the markup items
            numItems = len(item.markup)
            if numItems == 1:
                # the special case
                _item = item.markup[0]
                _item.indices = parentIndices
                _item.uvs = item.uvs
                r.createFace(building, _item.indices)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                verts = building.verts
                rs = self.renderState
                geometry.initRenderStateForDivs(rs, item)
                # a unit vector along U-axis (the horizontal axis)
                unitVector = (verts[parentIndices[1]] - verts[rs.indexLB]) / item.width
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        geometry.renderDivs(
                            self, building, item, unitVector,
                            0, numItems, 1,
                            rs
                        )
                        if symmetry:
                            geometry.renderDivs(
                                self, building, item, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                rs
                            )
                geometry.renderDivs(
                    self, building, item, unitVector,
                    0, numItems if symmetry else numItems-1, 1,
                    rs
                )
                if symmetry:
                    geometry.renderDivs(
                        self, building, item, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        rs
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                geometry.renderLastDiv(
                    self, item, lastItem,
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
    
    def setMaterialId(self, item, building, buildingPart, uvs, itemRenderer):
        facadePatternInfo = self.facadePatternInfo
        if self.initFacadePatternInfo:
            # reset <facadePatternInfo>
            for key in facadePatternInfo:
                facadePatternInfo[key] = 0
            # initalize <facadePatternInfo>
            for _item in item.markup:
                className = _item.__class__.__name__
                if className in facadePatternInfo:
                    facadePatternInfo[className] += 1
        # get a texture that fits to the Level markup pattern
        facadeTextureInfo = self.r.facadeTextureStore.getTextureInfo(
            building,
            buildingPart,
            facadePatternInfo
        )
        
        claddingMaterial = item.getStyleBlockAttrDeep("claddingMaterial")
        claddingTextureInfo = self.getCladdingTextureInfo(claddingMaterial, building)
        
        if facadeTextureInfo:
            materialId = self.getFacadeMaterialId(item, facadeTextureInfo, claddingTextureInfo)
            if itemRenderer.createFacadeMaterial(materialId, facadeTextureInfo, claddingTextureInfo, uvs):
                item.materialId = materialId
                item.materialData = facadeTextureInfo, claddingTextureInfo
            else:
                item.materialId = ""
        else:
            item.materialId = ""
    
    def renderLevelGroup(self, parentItem, levelGroup, indices, uvs):
        building = parentItem.building      
        face = self.r.createFace(building, indices)
        item = levelGroup.item
        if item:
            if item.materialId is None:
                if item.markup:
                    self.setMaterialId(
                        item,
                        building,
                        # getting building part
                        item.buildingPart if item.buildingPart else (
                            "level"
                            #"groundlevel" if levelGroup.singleLevel and not levelGroup.index1 else "level"
                        ),
                        uvs,
                        self
                    )
                else:
                    self.renderCladding(building, item, face, uvs)
            if item.materialId:
                facadeTextureInfo, claddingTextureInfo = item.materialData
                self.r.setUvs(
                    face,
                    item.geometry.getFinalUvs(
                        item,
                        self.getNumLevelsInFace(levelGroup),
                        facadeTextureInfo["numTilesU"],
                        facadeTextureInfo["numTilesV"]
                    ),
                    self.r.layer.uvLayerNameFacade
                )
                # set UV-coordinates for the cladding texture
                if not self.exportMaterials:
                    self.setCladdingUvs(item, face, claddingTextureInfo, uvs)
                self.setVertexColor(item, face)
            self.r.setMaterial(face, item.materialId)
        else:
            self.renderCladding(building, parentItem, face, uvs)