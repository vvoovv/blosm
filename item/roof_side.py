from . import Item


_className = "RoofSide"


class RoofSide(Item):
    
    def __init__(self):
        super().__init__()
        self.buildingPart = "roof_side"
        # slot index for the profile roofs
        self.slotIndex = 0
        # indices of <building.verts> that form the roof side
        self.indices = None
    
    @classmethod
    def getItem(cls, itemFactory, parent, indices, slotIndex):
        item = itemFactory.getItem(cls)
        item.init()
        item.parent = parent
        item.footprint = parent.footprint
        item.setStyleBlock()
        item.building = parent.building
        item.indices = indices
        item.slotIndex = slotIndex
        return item
    
    def setStyleBlock(self):
        # The logic for setting a style block for the roof side is the following:
        # (1) If <roofItem> has a markup (actually <styleBlocks>), then search the markup of
        #     the related style block for <RoofSide> style block.
        #     If the <RoofSide> style block wasn't found, then stop there and
        #     it means that there is no <RoofSide> style block.
        # (2) If <roofItem> does not have a markup (actually <styleBlocks>),
        #     then search for <RoofSide> style blocks in the markup
        #     (actually in <footprint.styleBlock.styleBlocks>) of the related footprint,
        #     also try to find them at the very top of the style definitions
        
        # <roofItem> is an instance of <item.roof_profile.RoofProfile
        roofItem = self.parent
        roofStyleBlock = roofItem.styleBlock
        if roofStyleBlock and _className in roofStyleBlock.styleBlocks:
            # Get <RoofSide> style blocks in in the markup (actually <styleBlocks>) of the roof item
            styleBlocks = roofStyleBlock.styleBlocks[_className]
        else:
            footprint = self.footprint
            # Find <RoofSide> style blocks in <markup> (actually in <footprint.styleBlock.styleBlocks>),
            # also try to find them at the very top of the style definitions
            styleBlocks = footprint.styleBlock.styleBlocks.get(
                _className,
                footprint.buildingStyle.styleBlocks.get(_className)
            )
        if styleBlocks:
            for styleBlock in styleBlocks:
                if self.evaluateCondition(styleBlock):
                    self.styleBlock = styleBlock
    
    @property
    def front(self):
        return True
    
    @property
    def back(self):
        return True
    
    def getStyleBlockAttr(self, attr):
        value = super().getStyleBlockAttr(attr) if self.styleBlock else None
        return value or self.parent.getStyleBlockAttr(attr)
    
    def getCladdingMaterial(self):
        return self.getStyleBlockAttr("roofCladdingMaterial")
    
    def getCladdingColor(self):
        return self.getStyleBlockAttr("roofCladdingColor")