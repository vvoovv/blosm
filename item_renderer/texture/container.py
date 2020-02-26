from .. import ItemRenderer
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast


class Container(ItemRenderer):
    """
    The base class for the item renderers Facade, Div, Layer, Bottom
    """
    
    def __init__(self, exportMaterials):
        super().__init__(exportMaterials)
        
    def getItemRenderer(self, item):
        return self.itemRenderers[item.__class__.__name__]
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            face = self.r.createFace(item.building, item.indices)
            self.renderCladding(item.building, item, face, item.uvs)
            return
            self.renderLevels(item)
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        parentIndices = item.indices
        geometry = item.geometry
        levelGroups = item.levelGroups
        levelGroups.init()
        # sanity check
        width = item.getWidthForVerticalArrangement()
        if width > item.width:
            item.valid = False
            return
        
        footprint = item.footprint
        building = item.building
        levelHeights = footprint.levelHeights
        
        # <indexBL> and <indexBR> are indices of the bottom vertices of an level item to be created
        # The prefix <BL> means "bottom left"
        indexBL = parentIndices[0]
        # The prefix <BR> means "bottom rights"
        indexBR = parentIndices[1]
        # <texVb> is the current V-coordinate for texturing the bottom vertices of
        # level items to be created out of <item>
        texVb = item.uvs[0][1]
        
        # treat the bottom
        if not footprint.minHeight:
            bottomHeight = item.getStyleBlockAttr("bottomHeight")
            if bottomHeight is None:
                bottomHeight = levelHeights.bottomHeight
            if bottomHeight:
                indexBL, indexBR, texVb = geometry.renderLevelGroup(
                    building, levelGroups.bottom, item, self.bottomRenderer, bottomHeight,
                    indexBL, indexBR,
                    texVb
                )
        
        # treat the level groups
        groups = levelGroups.groups
        numGroups = levelGroups.numActiveGroups
        minLevel = footprint.minLevel
        groupFound = not minLevel
        if numGroups > 1:
            for i in reversed(range(1, numGroups)):
                group = groups[i]
                if not groupFound and group.index1 <= minLevel <= group.index2:
                    groupFound = True
                if groupFound:
                    height = levelHeights.getLevelHeight(group.index1)\
                        if group.singleLevel else\
                        levelHeights.getHeight(group.index1, group.index2)
                    indexBL, indexBR, texVb = geometry.renderLevelGroup(
                        building, group, item, self.levelRenderer.getRenderer(group), height,
                        indexBL, indexBR,
                        texVb
                    )
        
        # the last level group
        geometry.renderLastLevelGroup(
            self, building, groups[0], item,
            indexBL, indexBR,
            texVb
        )
    
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
                # <indexLB> and <indexLT> are indices of the leftmost vertices of an item to be created
                # The prefix <LB> means "left bottom"
                indexLB = parentIndices[0]
                # The prefix <LT> means "left top"
                indexLT = parentIndices[-1]
                # a unit vector along U-axis (the horizontal axis)
                unitVector = (verts[parentIndices[1]] - verts[indexLB]) / item.width
                # <texUl> is the current U-coordinate for texturing the leftmost vertices of
                # items to be created out of <item>
                texUl = item.uvs[0][0]
                # <texVlt> is the current V-coordinate for texturing the top leftmost vertex of
                # items to be created out of <item>
                texVlt = item.uvs[-1][1]
                # <startIndex> is only used by geometry <TrapezoidChainedRv>
                startIndex = len(parentIndices) - 1
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        indexLB, indexLT, texUl, texVlt, startIndex = geometry.renderDivs(
                            self, building, item, unitVector,
                            0, numItems, 1,
                            indexLB, indexLT,
                            texUl, texVlt,
                            startIndex
                        )
                        if symmetry:
                            indexLB, indexLT, texUl, texVlt, startIndex = geometry.renderDivs(
                                self, building, item, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                indexLB, indexLT,
                                texUl, texVlt,
                                startIndex
                            )
                indexLB, indexLT, texUl, texVlt, startIndex = geometry.renderDivs(
                    self, building, item, unitVector,
                    0, numItems if symmetry else numItems-1, 1,
                    indexLB, indexLT,
                    texUl, texVlt,
                    startIndex
                )
                if symmetry:
                    indexLB, indexLT, texUl, texVlt, startIndex = geometry.renderDivs(
                        self, building, item, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        indexLB, indexLT,
                        texUl, texVlt,
                        startIndex
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                geometry.renderLastDiv(
                    self, item, lastItem,
                    indexLB, indexLT,
                    texUl, texVlt,
                    startIndex
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
    
    def renderLevelGroup(self, building, levelGroup, parentItem, indices, uvs):        
        face = self.r.createFace(building, indices)
        if levelGroup:
            item = levelGroup.item
            if item.materialId is None:
                if item.markup:
                    self.setMaterialId(
                        item,
                        building,
                        # getting building part
                        item.buildingPart if item.buildingPart else (
                            "groundlevel" if levelGroup.singleLevel and not levelGroup.index1 else "level"
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