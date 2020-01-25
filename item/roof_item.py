from . import Item

_className = "Roof"


class RoofItem(Item):
    
    def setStyleBlock(self):
        footprint = self.footprint
        # Find <Roof> style blocks in <markup> (actually in <self.styleBlock.styleBlocks>),
        # also try to find them at the very top of the style definitions
        styleBlock = footprint.styleBlock.styleBlocks.get(
            _className,
            footprint.buildingStyle.styleBlocks.get(_className)
        )
        if styleBlock:
            # a footprint can only have a single flat roof
            self.styleBlock = styleBlock[0]