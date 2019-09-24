from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast

from util import zAxis


class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def init(self, itemRenderers, globalRenderer):
        self.itemRenderers = itemRenderers
        self.r = globalRenderer
    
    def getItemRenderer(self, item):
        return self.itemRenderers[item.__class__.__name__]
    
    def renderMarkup(self, item):
        item.prepareMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            self.renderLevels(item)
        else:
            self.renderDivs(item)
        if not item.valid:
            return
    
    def renderLevels(self, item):
        # <r> is the global building renderer
        r = self.r
        parentIndices = item.indices
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
        
        prevIndex1 = parentIndices[0]
        prevIndex2 = parentIndices[1]
        index1 = len(building.verts)
        index2 = index1 + 1
        texU1, texV = item.uvs[0]
        texU2 = item.uvs[1][0]
        
        # treat the basement
        if not footprint.minHeight:
            basementHeight = item.getStyleBlockAttr("basementHeight")
            if basementHeight is None:
                basementHeight = levelHeights.basementHeight
            if basementHeight:
                prevIndex1, prevIndex2, index1, index2, texV = self.generateDiv(
                    building, levelGroups.basement, basementHeight, prevIndex1, prevIndex2, index1, index2,
                    texU1, texU2, texV
                )
                #basement = levelGroups.basement
                #if basement:
                #    basement.indices = indices
        
        # treat the level groups
        groups = levelGroups.groups
        numGroups = levelGroups.numActiveGroups
        minLevel = footprint.minLevel
        groupFound = not minLevel
        if numGroups > 1:
            for i in range(numGroups-1):
                group = groups[i]
                if not groupFound and group.index1 <= minLevel <= group.index2:
                    groupFound = True
                if groupFound:
                    height = levelHeights.getLevelHeight(group.index1)\
                        if group.singleLevel else\
                        levelHeights.getHeight(group.index1, group.index2)
                    prevIndex1, prevIndex2, index1, index2, texV = self.generateDiv(
                        building, group.item, height, prevIndex1, prevIndex2, index1, index2,
                        texU1, texU2, texV
                    )
        
        # the last level group
        indices = (prevIndex1, prevIndex2, parentIndices[2], parentIndices[3])
        texV2 = item.uvs[2][1]
        self.levelRenderer.render(
            building, groups[numGroups-1].item,
            indices,
            ( (texU1, texV), (texU2, texV), (texU2, texV2), (texU1, texV2) )
        )
    
    def renderDivs(self, item):
        # <r> is the global building renderer
        r = self.r
        building = item.building
        parentIndices = item.indices
        
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
                r.createFace(building, _item.indices, _item.uvs)
            else:
                numRepeats = item.numRepeats
                symmetry = item.symmetry
                verts = building.verts
                prevIndex1 = parentIndices[0]
                prevIndex2 = parentIndices[3]
                #self.v1 = verts[self.prevIndex1]
                #self.v2 = verts[self.prevIndex2]
                unitVector = (verts[parentIndices[1]] - verts[prevIndex1]) / item.width
                index1 = len(building.verts)
                index2 = index1 + 1
                # <texU> is the current U-coordinate for texturing
                # <texV1> and <texV2> are the lower and upper V-coordinates for texturing
                texU, texV1 = item.uvs[0]
                texV2 = item.uvs[3][1]
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                            building, item, unitVector,
                            0, numItems, 1,
                            prevIndex1, prevIndex2, index1, index2,
                            texU, texV1, texV2
                        )
                        if symmetry:
                            prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                                building, item, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1,
                                prevIndex1, prevIndex2, index1, index2,
                                texU, texV1, texV2
                            )
                prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                    building, item, unitVector,
                    0, numItems if symmetry else numItems-1, 1,
                    prevIndex1, prevIndex2, index1, index2,
                    texU, texV1, texV2
                )
                if symmetry:
                    prevIndex1, prevIndex2, index1, index2, texU = self.generateDivs(
                        building, item, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1,
                        prevIndex1, prevIndex2, index1, index2,
                        texU, texV1, texV2
                    )
                # process the last item
                lastItem = item.markup[0] if symmetry else item.markup[-1]
                texU2 = item.uvs[1][0]
                self.getItemRenderer(lastItem).render(
                    lastItem,
                    (prevIndex1, parentIndices[1], parentIndices[2], prevIndex2),
                    ( (texU, texV1), (texU2, texV1), (texU2, texV2), (texU, texV2) )
                )
        else:
            pass
        
    def generateDivs(self,
            building, item, unitVector, markupItemIndex1, markupItemIndex2, step,
            prevIndex1, prevIndex2, index1, index2, texU, texV1, texV2
        ):
        verts = building.verts
        v1 = verts[prevIndex1]
        v2 = verts[prevIndex2]
        for _i in range(markupItemIndex1, markupItemIndex2, step):
            _item = item.markup[_i]
            incrementVector = _item.width * unitVector
            v1 = v1 + incrementVector
            verts.append(v1)
            v2 = v2 + incrementVector
            verts.append(v2)
            texU2 = texU + _item.width
            self.getItemRenderer(_item).render(
                _item,
                (prevIndex1, index1, index2, prevIndex2),
                ( (texU, texV1), (texU2, texV1), (texU2, texV2), (texU, texV2) )
            )
            prevIndex1 = index1
            prevIndex2 = index2
            index1 = len(building.verts)
            index2 = index1 + 1
            texU = texU2
        return prevIndex1, prevIndex2, index1, index2, texU
    
    def generateDiv(self,
            building, item, height,
            prevIndex1, prevIndex2, index1, index2, texU1, texU2, texV
        ):
            verts = building.verts
            verts.append(verts[prevIndex1] + height*zAxis)
            verts.append(verts[prevIndex2] + height*zAxis)
            texV2 = texV + height
            self.levelRenderer.render(
                building, item,
                (prevIndex1, prevIndex2, index2, index1),
                ( (texU1, texV), (texU2, texV), (texU2, texV2), (texU1, texV2) )
            )
            prevIndex1 = index1
            prevIndex2 = index2
            index1 += 2
            index2 = index1 + 1
            return prevIndex1, prevIndex2, index1, index2, texV2