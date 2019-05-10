

class Library:
    """
    Library of style blocks with and without a namespace
    """
    
    def __init__(self):
        # the current style id
        self.styleId = -1
        # a place holder for style blocks without a namespace
        self.library = []
        self.numEntries = 0
        # a place holder for stytle blocks with a namespace
        self.libraryNS = {}
    
    def getStyleId(self):
        self.styleId += 1
        self.numEntries += 1
        self.library.append({})
        return self.styleId
    
    def addStyleBlock(self, defName, styleBlock, styleId):
        libraryEntry = self.library[styleId]
        className = styleBlock.__class__.__name__
        if not className in libraryEntry:
            libraryEntry[className] = {}
        libraryEntry[className][defName] = styleBlock
    
    def getStyleBlock(self, defName, styleBlockType, styleId):
        if styleId > self.numEntries:
            return None
        libraryEntry = self.library[styleId].get(styleBlockType)
        if not libraryEntry:
            return None
        return libraryEntry.get(defName)
    
    def addStyleBlockNS(self, defName, styleBlock, namespace):
        libraryEntry = self.libraryNS.get(namespace)
        if not libraryEntry:
            libraryEntry = {}
            self.libraryNS[namespace] = libraryEntry
        className = styleBlock.__class__.__name__
        if not className in libraryEntry:
            libraryEntry[className] = {}
        libraryEntry[className][defName] = styleBlock

    def getStyleBlockNS(self, defName, styleBlockType, namespace):
        if not namespace in self.libraryNS:
            return None
        libraryEntry = self.libraryNS[namespace].get(styleBlockType)
        if not libraryEntry:
            return None
        return libraryEntry.get(defName)


library = Library()