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
    
    def setRenderer(self, renderer):
        """
        Sets a renderer.
        For the 3D mode it's typically a Facade renderer.
        For the 2D mode it's a dedicated 3D renderer that creates footprints only
        """
        self.renderer = renderer
    
    def do(self, building, itemClass, style):
        itemStore = self.itemStore
        while itemStore.hasItems(itemClass):
            footprint = itemStore.getItem(itemClass)
            
            if footprint.styleBlock:
                # the footprint has been generated
                footprint.calculateFootprint()
            else:
                # the footprint is defined in the external data (e.g. OpenStreetMap)
                footprint.calculateStyling(style)
            # Check if have one or more footprints are defined in the markup definition,
            # it actually means, that those footprints are to be generated
            styleBlocks = footprint.styleBlock.styleBlocks.get("Footprint")
            if styleBlocks:
                for styleBlock in styleBlocks:
                    _footprint = Footprint.getItem(self.itemFactory, None, building, styleBlock)
                    _footprint.parent = footprint
                    itemStore.add(_footprint)
            volumeGenerator = self.volumeGenerators.get(
                footprint.getStyleBlockAttr("roofShape"),
                self.volumeGenerators[self.defaultRoofShape]
            )
            volumeGenerator.do(footprint, building, self.renderer)
    
    def setVolumeGenerators(self, data):
        #self.flatRoofMulti = RoofFlatMulti()
        self.volumeGenerators = {
            'flat': RoofFlat(data, self.itemStore, self.itemFactory),
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