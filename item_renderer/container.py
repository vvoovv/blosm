from grammar.arrangement import Horizontal


class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def renderMarkup(self, item):
        styleBlock = item.styleBlock
        # check if have levels or divs
        for _styleBlock in styleBlock.markup:
            if _styleBlock.isLevel:
                self.renderLevels(item)
            else:
                self.renderDivs(item)
            break
    
    def renderLevels(self, item, styleBlock):
        pass
    
    def renderDivs(self, item):
        # get the arrangement (horizontal or vertical) of the markup elements
        arrangement = item.getStyleBlockAttr("arrangement")
        if not arrangement:
            # the default arrangement of the markup elements
            arrangement = item.arrangement
        if arrangement is Horizontal:
            # get markup width and number of repeats
            item.calculateMarkupDivision()
        else:
            pass
        