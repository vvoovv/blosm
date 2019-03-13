from types import GeneratorType


class ItemStore:
    
    def __init__(self, referenceItems):
        self.skip = False
        store = {}
        self.store = store
        for referenceItem in referenceItems:
            if isinstance(referenceItem, tuple):
                referenceItem = referenceItem[0]
            # - A list of items
            # - Index of the current item
            # - The number of items
            store[referenceItem.__class__.__name__] = [[], 0, 0]
    
    def add(self, items, itemClass=None, numItems=0):
        if isinstance(items, GeneratorType):
            itemsEntry = self.store[itemClass.__name__]
            itemsEntry[0].extend(items)
            itemsEntry[2] += numItems
        else:
            # single item
            itemsEntry = self.store[items.__class__.__name__]
            itemsEntry[0].append(items)
            itemsEntry[2] += 1
    
    def hasItems(self, itemClass):
        itemsEntry = self.store[itemClass.__name__]
        return itemsEntry[2] and itemsEntry[1] < itemsEntry[2]
    
    def getItem(self, itemClass):
        """
        Returns the current item or None if no item is left
        """
        itemsEntry = self.store[itemClass.__name__]
        if itemsEntry[2] and itemsEntry[1] < itemsEntry[2]:
            item = itemsEntry[0][itemsEntry[1]]
            itemsEntry[1] += 1
        else:
            # no item is left
            item = None
        return item
    
    def clear(self):
        for itemClass in self.store:
            itemsEntry = self.store[itemClass]
            itemsEntry[0].clear()
            itemsEntry[1] = 0
            itemsEntry[2] = 0