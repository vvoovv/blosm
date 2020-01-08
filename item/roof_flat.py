from . import Item

_className = "Roof"


class RoofFlat(Item):
    
    @classmethod
    def getItem(cls, itemFactory, parent, indices):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent
        item.setStyleBlock()
        item.building = parent.building
        item.indices = indices
        return item
    
    def setStyleBlock(self):
        footprint = self.footprint
        # Find <Facade> style blocks in <markup> (actually in <self.styleBlock.styleBlocks>),
        # also try to find them at the very top of the style definitions
        styleBlock = footprint.styleBlock.styleBlocks.get(
            _className,
            footprint.buildingStyle.styleBlocks.get(_className)
        )
        if styleBlock:
            # a footrpint can only have a single flat roof
            self.styleBlock = styleBlock[0]