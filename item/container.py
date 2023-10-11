import math
from . import Item
from grammar.arrangement import Horizontal, Vertical
from grammar.symmetry import MiddleOfLast


class ItemSize:
    
    def __init__(self):
        self.fixed = 0.
        self.flex = 0.
    
    def init(self):
        self.fixed = 0.
        self.flex = 0.


class Container(Item):
    
    def __init__(self, parent, footprint, facade, styleBlock):
        super().__init__(parent, footprint, facade, styleBlock)
        # a Python list to store markup items
        self.markup = []
        # The meaning of <self.width> and <self.height> for the items derived from <Container>
        # is described below.
        #
        # (*) If a dimension (either <self.width> or <self.height>) is <None>, then the markup elements
        # are not repeated within the item boundaries along the dimension.
        #
        # (*) If a dimension (either <self.width> or <self.height>) is NOT <None> and
        # the attribute <repeat> is equal to <True> in the related style block or
        # in the default settings of the item, then the markup elements are repeated to fit to
        # the the given item boundaries along the dimension.
        self.width = None
        self.height = None
        
        # the number of repeats if the markup items if the item has a fixed width
        self.numRepeats = 1
        
        # the default arrangement of markup items
        self.arrangement = Horizontal
        
        # Do we have a symmetry for the markup items?
        # Allowed values: None or symmetry.MiddleOfLast or symmetry.RightmostOfLast
        self.symmetry = None
        
        # indices of vertices in the <verts> and <bmVerts>
        self.indices = None
        self.uvs = None
        
        # an item renderer might need some data related to the material with <self.materialId>
        self.materialData = None
    
    def getWidth(self, globalRenderer):
        if self.styleBlock.markup and not self.markup:
            self.prepareMarkupItems()
        
        if self.markup:
            return 0.
        else:
            return globalRenderer.itemRenderers[self.__class__.__name__].getTileWidthM(self)
    
    def prepareMarkupItems(self):
        """
        Get items for the markup style blocks
        """
        if self.markup:
            return
        
        if self.styleBlock.markup:
            self.markup.extend(
                _styleBlock.getItem(self)\
                    for _styleBlock in self.styleBlock.markup if self.evaluateCondition(_styleBlock)
            )
            
            # check if have levels in the markup
            if self.styleBlock.markup[0].isLevel:
                # the arrangement of the Level items is always vertical 
                self.arrangement = Vertical
            else:
                # set the arrangement (horizontal or vertical) of the markup items
                arrangement = self.getStyleBlockAttr("arrangement")
                if arrangement:
                    self.arrangement = arrangement
            
            if self.arrangement == Vertical:
                # if the item is located in the left corner of <self.facade>, mark
                # all items in <self.markup> that they are located in the left corner too.
                if self.cornerL:
                    for item in self.markup:
                        item.cornerL = True
                # if the item is located in the right corner of <self.facade>, mark
                # all items in <self.markup> that they are located in the right corner too.
                if self.cornerR:
                    for item in self.markup:
                        item.cornerR = True
            else: # self.arrangement == Horizontal
                # if the item is located in the left corner of <self.facade>, mark
                # the leftmost item in <self.markup> that is located in the left corner too.
                if self.cornerL:
                    self.markup[0].cornerL = True
                # if the item is located in the right corner of <self.facade>, mark
                # the rightmost item in <self.markup> that is located in the right corner too.
                if self.cornerR:
                    self.markup[-1].cornerR = True
                
        # check if have symmetry for the markup items
        symmetry = self.getStyleBlockAttr("symmetry")
        if symmetry:
            self.symmetry = symmetry
    
    def calculateMarkupDivision(self, globalRenderer):
        markup = self.markup
        
        totalFixedWidth = 0.
        totalFlexWidth = 0.
        totalRelativeWidth = 0.
        # Items that have flexible number of child items. The number of child items is
        # defined be the resulting width of the item
        flexibleChildrenItems = []
        
        # <repeat> is equal to <True> by default
        repeat = bool( self.getStyleBlockAttr("repeat") )
        
        # iterate through the markup items
        for item in markup:
            width = item.getStyleBlockAttr("width")
            if width:
                item.width = width
                totalFixedWidth += width
            else:
                relativeWidth = item.getStyleBlockAttr("relativeWidth")
                if relativeWidth:
                    item.relativeWidth = relativeWidth
                    totalRelativeWidth += relativeWidth
                else:
                    # We check if <item> is a container (i.e. level or div)
                    if item.isContainer and not item.styleBlock.markup:
                        flexibleChildrenItems.append(item)
                    else:
                        width = item.getWidth(globalRenderer)
                        item.hasFlexWidth = True
                        item.width = width
                        totalFlexWidth += width
        
        # treat the case with the symmetry (currently commented out)
        """
        symmetry = self.getStyleBlockAttr("symmetry")
        if symmetry:
            self.symmetry = symmetry
            if totalFixedWidth:
                totalFixedWidth *= 2.
            if totalFlexWidth:
                totalFlexWidth *= 2.
            if totalRelativeWidth:
                totalRelativeWidth *= 2.
            # the special case if <symmetry> is <MiddleOfLast>
            if symmetry is MiddleOfLast:
                middleItem = markup[-1]
                if middleItem.width:
                    if middleItem.hasFlexWidth:
                        totalFlexWidth -= middleItem.width
                    else:
                        totalFixedWidth -= middleItem.width
                else:
                    totalRelativeWidth -= middleItem.relativeWidth
        """
        
        totalNonRelativeWidth = totalFixedWidth+totalFlexWidth
        
        # perform sanity check
        if (totalRelativeWidth and not 0. < totalRelativeWidth <= 1.) or\
            (
                self.width and\
                (
                    totalNonRelativeWidth > self.width or\
                    (
                        totalRelativeWidth and totalNonRelativeWidth and\
                        (totalRelativeWidth == 1. or totalNonRelativeWidth/(1-totalRelativeWidth) > self.width)
                    )
                )
            ):
            self.valid = False
            return
        
        # process the results of the first iteration through the markup items
        
        # treat the case with repeats first
        if repeat:
            if totalRelativeWidth:
                if totalNonRelativeWidth:
                    # width of a single markup pattern without any repeats
                    width = totalNonRelativeWidth / (1. - totalRelativeWidth)
                else:
                    # All markup items has relative width
                    # Calculate the width estimate for the markup items, it's also the width of
                    # a single markup pattern without any repeats
                    width = sum(item.getWidth() for item in markup) / totalRelativeWidth
                    # sanity check
                    if width > self.width:
                        self.valid = False
                        return
            else:
                # there are no items with the relative width
                width = totalNonRelativeWidth
            
            numRepeats = math.floor(self.width / width)
            self.numRepeats = numRepeats
            factor = self.width/numRepeats/width
            if numRepeats > 1:
                # the corrected and final width of a single markup pattern without any repeats
                width *= factor
            else:
                width = self.width
            
            # update the widths of the markup items to fit them to <width>
            for item in markup:
                if item.relativeWidth:
                    item.width = item.relativeWidth * width
                elif item.hasFlexWidth:
                    item.width *= factor
            if totalFixedWidth:
                # distribute the width <factor>*<totalFixedWidth> to the left and right margins
                pass # TODO
        elif totalRelativeWidth:
            if totalNonRelativeWidth:
                # Calculate width of the markup items using
                # <totalFixedWidth>, <totalFlexWidth> and <totalRelativeWidth>
                if self.width:
                    # <_width> is the total width of the markup items with the relative width
                    _width = totalRelativeWidth * self.width
                    # sanity check
                    if _width > (self.width - totalNonRelativeWidth):
                        self.valid = False
                        return
                    extraWidth = self.width - _width - totalNonRelativeWidth
                    if totalFlexWidth:
                        _totalFlexWidth = totalFlexWidth + extraWidth
                        # distribute the excessive width among the markup items with the flexible width
                        for item in markup:
                            if item.hasFlexWidth:
                                item.width *= _totalFlexWidth/totalFlexWidth
                    else:
                        # distribute the excessive width to the left and right margins
                        pass # TODO
                else:
                    # <width> is the total width of all markup elements
                    width = totalNonRelativeWidth / (1. - totalRelativeWidth)
                    self.width = width
                    _width = width - totalNonRelativeWidth
                # The solution for the case <totalRelativeWidth> is grater than or equal to 1 is 
                # to assume that <totalRelativeWidth> corresponds to <_width>
                for item in markup:
                    if item.relativeWidth:
                        item.width = item.relativeWidth * _width / totalRelativeWidth
            else:
                # all markup items has relative width
                if self.width:
                    width = self.width
                else:
                    # Calculate width estimate for the markup items
                    width = sum(item.getWidth() for item in markup)
                    self.width = width
                # set width for each item:
                for item in markup:
                    item.width = item.relativeWidth*width/totalRelativeWidth
        else:
            # there are no items with the relative width
            if totalNonRelativeWidth < self.width:
                if flexibleChildrenItems:
                    # Total width for the items in <flexibleChildrenItems>.
                    # The items in <noWidthItems> have the same with by design
                    _width = (self.width - totalNonRelativeWidth)/len(flexibleChildrenItems)
                    for item in flexibleChildrenItems:
                        item.width = _width
                elif totalFlexWidth:  
                    factor = (self.width - totalFixedWidth)/totalFlexWidth
                    # distribute the excessive width among the markup items with the flexible width
                    for item in markup:
                        if item.hasFlexWidth:
                            item.width *= factor
                else:
                    # distribute the excessive width to the left and right margins
                    pass # TODO
            else:
                self.valid = False
                return
        # always return the total width of all markup elements
        return self.width
    
    def getWidthForVerticalArrangement(self):
        return max(item.getWidth() for item in self.markup)
            