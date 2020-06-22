from . import Item

_className = "Roof"


class RoofItem(Item):
    
    def __init__(self):
        super().__init__()
        self.buildingPart = "roof"
    
    def setStyleBlock(self):
        footprint = self.footprint
        # Find <Roof> style blocks in <markup> (actually in <self.styleBlock.styleBlocks>),
        # also try to find them at the very top of the style definitions
        styleBlocks = footprint.styleBlock.styleBlocks.get(
            _className,
            footprint.buildingStyle.styleBlocks.get(_className)
        )
        if styleBlocks:
            for styleBlock in styleBlocks:
                if self.evaluateCondition(styleBlock):
                    self.styleBlock = styleBlock
                    if styleBlock.markup:
                        self.prepareMarkupItems()
                    break

    def prepareMarkupItems(self):
        """
        Get items for the markup style blocks
        """
        self.markup.extend(
            _styleBlock.getItem(self.itemFactory, self)\
                for _styleBlock in self.styleBlock.markup if self.evaluateCondition(_styleBlock)
        )

    def getCladdingMaterial(self):
        return self.getStyleBlockAttr("roofCladdingMaterial")
    
    def getCladdingColor(self):
        return self.getStyleBlockAttr("roofCladdingColor")