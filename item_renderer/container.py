from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast, RightmostOfLast

from util import zAxis


class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        # the variables below are used to generate Divs
        self.prevIndex1 = 0
        self.prevIndex2 = 0
        self.index1 = 0
        self.index2 = 0
        self.v1 = None
        self.v2 = None
        # the current U-coordinate for texturing
        self.texU = 0.
        # the lower and upper V-coordinates for texturing
        self.texV1 = 0.
        self.texV2 = 0.
    
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
        verts = building.verts
        levelHeights = footprint.levelHeights
        
        self.prevIndex1 = parentIndices[0]
        self.prevIndex2 = parentIndices[1]
        self.v1 = verts[self.prevIndex1]
        self.v2 = verts[self.prevIndex2]
        self.index1 = len(verts)
        self.index2 = self.index1 + 1
        
        # treat the basement
        if not footprint.minHeight:
            basementHeight = levelHeights.basementHeight
            if basementHeight:
                building.appendBmVerts(2)
                verts.append(self.v1 + basementHeight*zAxis)
                verts.append(self.v2 + basementHeight*zAxis)
                indices = (self.prevIndex1, self.prevIndex2, self.index2, self.index1)
                self.prevIndex1 = self.index1
                self.prevIndex2 = self.index2
                basement = levelGroups.basement
                if basement:
                    basement.indices = indices
                r.createFace(building, indices, None)
        
        # treat the level groups
        groups = levelGroups.groups
        numGroups = levelGroups.numActiveGroups
        if numGroups > 1:
            for i in range(numGroups-1):
                group = groups[i]
        
        # the last level group
        indices = (self.prevIndex1, self.prevIndex2, parentIndices[2], parentIndices[3])
        groups[-1].indices = indices
        r.createFace(building, indices, None)
    
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
                building.appendBmVerts(
                    2*(
                        (numRepeats*(2*numItems-1) - 1) if symmetry is MiddleOfLast else (
                            (numRepeats*2*numItems-1) if symmetry is RightmostOfLast\
                            else (numRepeats*numItems-1)
                        )
                    )
                )
                verts = building.verts
                self.prevIndex1 = parentIndices[0]
                self.prevIndex2 = parentIndices[3]
                self.v1 = verts[self.prevIndex1]
                self.v2 = verts[self.prevIndex2]
                unitVector = (verts[parentIndices[1]] - self.v1) / item.width
                self.index1 = len(verts)
                self.index2 = self.index1 + 1
                self.texU, self.texV1 = item.uvs[0]
                self.texV2 = item.uvs[3][1]
                
                # Generate Div items but the last one;
                # the special case is when a symmetry is available
                if numRepeats>1:
                    for _ in range(numRepeats-1):
                        self.generateDivs(
                            building, item, unitVector,
                            0, numItems, 1
                        )
                        if symmetry:
                            self.generateDivs(
                                building, item, unitVector,
                                numItems-2 if symmetry is MiddleOfLast else numItems-1, -1, -1
                            )
                self.generateDivs(
                    building, item, unitVector,
                    0, numItems if symmetry else numItems-1, 1
                )
                if symmetry:
                    self.generateDivs(
                        building, item, unitVector,
                        numItems-2 if symmetry is MiddleOfLast else numItems-1, 0, -1
                    )
                # process the last item
                _item = item.markup[-1]
                _item.indices = (self.prevIndex1, parentIndices[1], parentIndices[2], self.prevIndex2)
                texU = item.uvs[1][0]
                r.createFace(
                    building,
                    _item.indices,
                    (
                        (self.texU, self.texV1), (texU, self.texV1), (texU, self.texV2), (self.texU, self.texV2)
                    )
                )
        else:
            pass
        
    def generateDivs(self, building, item, unitVector, itemIndex1, itemIndex2, step):
        verts = building.verts
        texU = self.texU
        for _i in range(itemIndex1, itemIndex2, step):
            _item = item.markup[_i]
            incrementVector = _item.width * unitVector
            self.v1 = self.v1 + incrementVector
            verts.append(self.v1)
            self.v2 = self.v2 + incrementVector
            verts.append(self.v2)
            _item.indices = (self.prevIndex1, self.index1, self.index2, self.prevIndex2)
            self.prevIndex1 = self.index1
            self.prevIndex2 = self.index2
            self.index1 += 2
            self.index2 += 2
            texU = self.texU + _item.width
            self.r.createFace(
                building,
                _item.indices,
                (
                    (self.texU, self.texV1), (texU, self.texV1), (texU, self.texV2), (self.texU, self.texV2)
                )
            )
            self.texU = texU