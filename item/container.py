import math
from . import Item
from grammar.arrangement import Horizontal, Vertical


class ItemSize:
    
    def __init__(self):
        self.fixed = 0.
        self.flex = 0.
    
    def init(self):
        self.fixed = 0.
        self.flex = 0.


class Container(Item):
    
    def __init__(self):
        super().__init__()
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
                
        # do we need to repeat the markup items, if the item has a fixed width?
        self.repeat = True
        # The number of repeats if the markup items if the item has a fixed width and
        # <self.repeat> is True
        self.numRepeats = 1
        
        # the default arrangement of markup items
        self.arrangement = Horizontal
    
    def init(self):
        super().init()
        self.markup.clear()
        self.width = None
        self.height = None
        self.repeat = True
        self.arrangement = Horizontal
    
    def getWidth(self):
        if not self.markup:
            self.getMarkupItems()
        return self.width or (
            self.calculateMarkupDivision() if self.arrangement is Horizontal else\
                self.getWidthForVerticalArrangement()
        )
    
    def getMarkupItems(self):
        """
        Get items for the markup style blocks
        """
        self.markup.extend(
            _styleBlock.getItem(self.itemFactory, self)\
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
    
    def calculateMarkupDivision(self):
        markup = self.markup
        
        totalFixedWidth = 0.
        totalFlexWidth = 0.
        totalRelativeWidth = 0.
        
        if self.width:
            repeat = self.getStyleBlockAttr("repeat")
            if not repeat is None:
                self.repeat = bool(repeat)
        
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
                    # No width is given in the style block.
                    # So we calculate the width estimate
                    width = item.getWidth()
                    item.width = width
                    item.hasFlexWidth = True
                    totalFlexWidth += width
        
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
        if self.width and self.repeat:
            if totalRelativeWidth:
                if totalNonRelativeWidth:
                    # width of a single markup pattern without any repeats
                    width = totalRelativeWidth * totalNonRelativeWidth / (1. - totalRelativeWidth)
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
                # the corrected and final width of a single markup patter without any repeats
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
                    _width = self.width - totalNonRelativeWidth
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
            if self.width:
                if totalNonRelativeWidth < self.width:
                    extraWidth = self.width - totalNonRelativeWidth
                    if totalFlexWidth:
                        # distribute the excessive width among the markup items with the flexible width
                        for item in markup:
                            if item.hasFlexWidth:
                                _width = totalFlexWidth + extraWidth
                                item.width *= _width/totalFlexWidth
                    else:
                        # distribute the excessive width to the left and right margins
                        pass # TODO
            else:
                self.width = totalNonRelativeWidth
        # always return the total width of all markup elements
        return self.width
    
    def getWidthForVerticalArrangement(self):
        return max(item.getWidth() for item in self.markup)
            