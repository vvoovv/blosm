from grammar import Grammar


class StyleStore:
    
    def __init__(self):
        self.styles = {}
    
    def add(self, styleName, style):
        style = Grammar(style)
        self.styles[styleName] = style
    
    def get(self, styleName):
        return self.styles[styleName]