from . import Item
from grammar.arrangement import Horizontal


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
        
        # the default arrangement of markup items
        self.arrangement = Horizontal
    
    def init(self):
        super().init()
        self.markup.clear()
        self.width = None
        self.height = None
    
    def getWidth(self):
        return self.calculateMarkupDivision()
    
    def calculateMarkupDivision(self):
        markup = self.markup
        if not markup:
            # get items for the markup style blocks
            markup.extend(
                _styleBlock.getItem(self.itemFactory, self)\
                    for _styleBlock in self.styleBlock.markup if self.evaluateCondition(_styleBlock)
            )
        
        totalFixedWidth = 0.
        totalFlexWidth = 0.
        totalRelativeWidth = 0.
        
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
                    item.hasFlexWidth = True
                    totalFlexWidth += width
        
        totalNonRelativeWidth = totalFixedWidth+totalFlexWidth
        
        # perform sanity check
        if self.width and totalNonRelativeWidth > self.width:
            self.valid = False
            return
        # process the results of the first interation through the markup items
        if totalRelativeWidth:
            if totalNonRelativeWidth:
                # Calculate width of the markup items using
                # <totalFixedWidth>, <totalFlexWidth> and <totalRelativeWidth>
                if self.width:
                    # <_width> is the total width of the markup items
                    # the relative width
                    _width = self.width - totalNonRelativeWidth
                else:
                    if totalRelativeWidth >= 1.:
                        # Something is wrong: <totalRelativeWidth> must be less than 1
                        # Total width of all markup items is not set, so
                        # we calculate the width estimate for the markup items with the relative width
                        _width = sum(
                            item.getWidth() for item in markup if item.relativeWidth
                        )
                        self.width = totalNonRelativeWidth + _width
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
                # All markup items has relative width
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
                        # distribute the excessive width in the left and right margins
                        pass # TODO
            else:
                self.width = totalNonRelativeWidth
        # always return the total width of all markup elements
        return self.width