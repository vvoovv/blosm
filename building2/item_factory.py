

class ItemFactory:
    """
    The first item 
    """
    
    numItemsIncreaseStep = 20
    
    def __init__(self, referenceItems, defaultNumItems=20):
        items = {}
        self.items = items
        for referenceItem in referenceItems:
            if isinstance(referenceItem, tuple):
                referenceItem, numItems = referenceItem
            else:
                numItems = defaultNumItems
            # set item factory to be used in <item.calculateMarkupDivision(..)>
            referenceItem.itemFactory = self
            # A list:
            # - a list of items
            # - the index of the current item (i.e. used the last time)
            # - number of items
            _items = [referenceItem]
            _items.extend( referenceItem.clone() for _ in range(1, numItems) )
            items[referenceItem.__class__.__name__] = [_items, 1, numItems]
    
    def getItem(self, itemClass):
        itemsEntry = self.items[itemClass.__name__]
        items, itemIndex, numItems = itemsEntry
        if itemIndex==numItems:
            referenceItem = items[0]
            items.extend(referenceItem.clone() for _ in range(ItemFactory.numItemsIncreaseStep))
            itemsEntry[2] += ItemFactory.numItemsIncreaseStep
        itemsEntry[1] += 1
        return items[itemIndex]