

class Item:
    
    def __init__(self):
        # For example, a parent for a facade is a footprint
        self.parent = None
        # A style block (an instance of grammar.Item) that defines the style
        # for the item inside a markup definition.
        # Not all items (e.g. Footprint, Facade) can be part of a markup
        self.markupStyle = None
        self.calculatedStyle = {}
    
    def clone(self):
        item = self.__class__()
        return item
    
    def calculateStyle(self, styleDefs):
        """
        Calculates a specific style for the item out of the set of style definitions <styleDefs>
        
        Args:
            styleDefs (grammar.Grammar): a set of style definitions
        """
        pass    