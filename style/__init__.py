from grammar import Grammar


class StyleStore:
    
    def __init__(self):
        self.styles = {}
    
    def add(self, styleName, style):
        style = Grammar(defs=style)
        self.styles[styleName] = style