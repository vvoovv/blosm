from .library import library


class Item:
    
    def __init__(self, defName, use, markup, condition, attrs):
        self.defName = defName
        self.use = use
        self.markup = markup
        self.condition = condition
        self.attrs = attrs
    
    def init(self):
        self.initAttrs()
        if self.markup:
            for styleBlock in self.markup:
                self.setParent(styleBlock)
    
    def setParent(self, styleBlock):
        styleBlock.parent = self
    
    def build(self, styleId):
        if self.use:
            self.styleId = styleId
            self._applyUse()
        if self.markup:
            self.buildMarkup(styleId)
    
    def buildMarkup(self, styleId):
        for styleBlock in self.markup:
            styleBlock.build(styleId)
    
    def _applyUse(self):
        markup = self.markup
        # note the reversed iterator
        for defName in reversed(self.use):
            _styleBlock = library.getStyleBlock(
                defName, self.__class__.__name__, self.styleId
            )
            if _styleBlock:
                for attr in _styleBlock.attrs:
                    if not attr in self.attrs:
                        self.attrs[attr] = _styleBlock.attrs[attr]
                if not markup:
                    markup = _styleBlock.markup
        # set the markup if it was defined in a definition style block referenced in <self.use>
        if not self.markup and markup:
            self.markup = markup
    
    def __contains__(self, attr):
        return attr in self.attrs


class Grammar:
    """
    The top level element for the building style
    """
    
    def __init__(self, styleBlocks):
        # A placeholder the style blocks without a name, located on the very top of the hierachy.
        # Besides a Footprint, those style blocks can be of the types Facade,
        # RoofSide, Ridge, i.e. the items generated through volume extrusion
        self.styleBlocks = {}
        self.meta = None
        self.build(styleBlocks)
    
    def build(self, inputStyleBlocks):
        styleBlocks = self.styleBlocks
        
        styleId = library.getStyleId()
        
        # style block with <defName> (i.e. style definitions)
        defStyleBlocks = []
        # ordinary style blocks (without <defName>)
        normalStyleBlocks = []
        
        for styleBlock in inputStyleBlocks:
            className = styleBlock.__class__.__name__
            if className == "Meta":
                self.meta = styleBlock
            elif styleBlock.defName:
                library.addStyleBlock(styleBlock.defName, styleBlock, styleId)
                defStyleBlocks.append(styleBlock)
            else:
                if not className in styleBlocks:
                    styleBlocks[className] = []
                styleBlocks[className].append(styleBlock)
                styleBlock.parent = None
                normalStyleBlocks.append(styleBlock)
        
        for styleBlock in defStyleBlocks:
            styleBlock.build(styleId)
        
        for styleBlock in normalStyleBlocks:
            styleBlock.build(styleId)

    def setParent(self, item):
        item.parent = None


class Meta:
    
    def __init__(self, **attrs):
        self.attrs = attrs