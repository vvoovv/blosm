from grammar.arrangement import Horizontal, Vertical


class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def renderMarkup(self, item):
        item.getMarkupItems()
        
        if item.styleBlock.markup[0].isLevel:
            self.renderLevels(item)
        else:
            self.renderDivs(item)
    
    def renderLevels(self, item):
        pass
    
    def renderDivs(self, item):
        if item.arrangement is Horizontal:
            # get markup width and number of repeats
            item.calculateMarkupDivision()
        else:
            pass
        