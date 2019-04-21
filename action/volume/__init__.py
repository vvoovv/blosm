from .. import Action
from item.footprint import Footprint


# import roof generators
from .roof_flat import RoofFlat
#from .roof_gabled import RoofGabled


class Volume(Action):
    """
    This action creates a building volume out of its footprint
    """
    
    defaultRoofShape = "flat"
    
    def __init__(self, app, data, itemStore, itemFactory):
        super().__init__(app, data, itemStore, itemFactory)
        self.setVolumeGenerators(data)
    
    def do(self, building, itemClass, styleDefs):
        itemStore = self.itemStore
        while itemStore.hasItems(itemClass):
            item = itemStore.getItem(itemClass)
            item.calculateStyle(styleDefs)
            if not item.element:
                item.calculateFootprint()
            if item.markupStyle:
                # If one or more footprints are defined in the markup definition,
                # it actually means, that those footprints are to be generated
                for style in self.markupStyle.markup:
                    item = Footprint.getItem(self.itemFactory, None, style)
                    item.parent = self
                    itemStore.add(item)
            # now time to create the building volume itself
            style = item.calculatedStyle
            volumeGenerator = self.volumeGenerators.get(
                style["roofShape"],
                self.volumeGenerators[self.defaultRoofShape]
            )
            volumeGenerator.do(item, style, building)
    
    def setVolumeGenerators(self, data):
        #self.flatRoofMulti = RoofFlatMulti()
        self.volumeGenerators = {
            'flat': RoofFlat(data),
            #'gabled': RoofGabled()# RoofProfile(gabledRoof),
            #'pyramidal': RoofPyramidal(),
            #'skillion': RoofSkillion(),
            #'hipped': RoofHipped(),
            #'dome': RoofMesh("roof_dome"),
            #'onion': RoofMesh("roof_onion"),
            #'round': RoofProfile(roundRoof),
            #'half-hipped': RoofHalfHipped(),
            #'gambrel': RoofProfile(gambrelRoof),
            #'saltbox': RoofProfile(saltboxRoof),
            #'mansard': RoofMansard()
        }