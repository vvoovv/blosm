from .. import Action


class Volume(Action):
    """
    This action creates a building volume out of its footprint
    """
    
    def do(self, building, itemClass, styleDefs):
        itemStore = self.itemStore
        while itemStore.hasItems(itemClass):
            item = itemStore.getItem(itemClass)
            item.calculateStyle(styleDefs)
            s = 0