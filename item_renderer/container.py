

class Container:
    """
    The base class for the item renderers Facade, Div, Layer, Basement
    """
    
    def __init__(self):
        pass
    
    def init(self):
        pass
    
    def renderMarkup(self, facade, styleBlock):
        # check if have levels or divs
        for _styleBlock in styleBlock.markup:
            if _styleBlock.isLevel:
                self.renderLevels(facade, styleBlock)
            else:
                self.renderDivs(facade, styleBlock)
            break
    
    def renderLevels(self, facade, styleBlock):
        pass
    
    def renderDivs(self, facade, styleBlock):
        pass