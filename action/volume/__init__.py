from .. import Action
from item.footprint import Footprint
from renderer import Renderer

# import roof generators
from .roof_flat import RoofFlat
from .roof_flat_multi import RoofFlatMulti
from .roof_generatrix import RoofGeneratrix
from .roof_profile import RoofProfile, roofDataGabled, roofDataRound, roofDataGambrel, roofDataSaltbox


class Volume(Action):
    """
    This action creates a building volume out of its footprint
    """
    
    defaultRoofShape = "flat"
    
    def __init__(self, app, data, itemStore, itemFactory, itemRenderers):
        super().__init__(app, data, itemStore, itemFactory)
        if itemRenderers:
            self.setVolumeGenerators(data, itemRenderers)
    
    def setRenderer(self, renderer):
        """
        Sets a renderer.
        For the 3D mode it's typically a Facade renderer.
        For the 2D mode it's a dedicated 3D renderer that creates footprints only
        """
        self.renderer = renderer
    
    def prepareFootprint(self, footprint, building, buildingStyle):
        footprint.buildingStyle = buildingStyle
        
        if footprint.styleBlock:
            # the footprint has been generated
            footprint.calculateFootprint()
        else:
            # the footprint is defined in the external data (e.g. OpenStreetMap)
            footprint.calculateStyling()
        # Check if have one or more footprints are defined in the markup definition,
        # it actually means, that those footprints are to be generated
        styleBlocks = footprint.styleBlock.styleBlocks.get("Footprint")
        if styleBlocks:
            for styleBlock in styleBlocks:
                _footprint = Footprint.getItem(self.itemFactory, None, building, styleBlock)
                _footprint.parent = footprint
                _footprint.buildingStyle = buildingStyle
                self.itemStore.add(_footprint)
        return footprint
    
    def do(self, building, itemClass, buildingStyle, globalRenderer):
        itemStore = self.itemStore
        while itemStore.hasItems(itemClass):
            footprint = itemStore.getItem(itemClass)
            self.prepareFootprint(footprint, building, buildingStyle)
            
            element = footprint.element
            if element.t is Renderer.multipolygon:
                # check if the multipolygon has holes
                if element.hasInner():
                    self.volumeGeneratorMultiFlat.do(footprint)
                else:
                    # That's a quite rare case
                    # We treat each polygon of the multipolygon as a single polygon
                    # <footprint> won't be used in this case, a new footprint will be created
                    # for each polygon of the multipolygon
                    
                    # A note.
                    # If <building.footprint> is set, for example, in a <building.area()> call,
                    # we don't need to care about it in any way. Since <building.footprint>
                    # won't be used. A new footprint will be created for each polygon of
                    # the multipolygon.
                    
                    # overrides to pretend than <element> is a polygon
                    element.t = Renderer.polygon
                    ls = element.ls
                    for _l in ls:
                        element.ls = _l
                        footprint = Footprint.getItem(self.itemFactory, element, building)
                        self.prepareFootprint(footprint, building, buildingStyle)
                        self.generateVolume(
                            footprint,
                            element.getLinestringData(_l, self.data)
                        )
                    element.ls = ls
            else:
                self.generateVolume(footprint, element.getData(self.data))
    
    def generateVolume(self, footprint, coords):
        volumeGenerator = self.volumeGenerators.get(
            footprint.getStyleBlockAttr("roofShape"),
            self.volumeGenerators[Volume.defaultRoofShape]
        )
        volumeGenerator.do(footprint, coords)
    
    def setVolumeGenerators(self, data, itemRenderers):
        facadeRenderer = itemRenderers["Facade"]
        self.volumeGenerators = {
            'flat': RoofFlat(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofFlat"]),
            'gabled': RoofProfile(roofDataGabled, data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofProfile"]),
            'pyramidal': RoofGeneratrix(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofPyramidal"]),
            #'skillion': RoofSkillion(),
            #'hipped': RoofHipped(),
            'dome': RoofGeneratrix(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofDome"]),
            'half-dome': RoofGeneratrix(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofHalfDome"]),
            'onion': RoofGeneratrix(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofOnion"]),
            'round': RoofProfile(roofDataRound, data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofProfile"]),
            #'half-hipped': RoofHalfHipped(),
            'gambrel': RoofProfile(roofDataGambrel, data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofProfile"]),
            'saltbox': RoofProfile(roofDataSaltbox, data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofProfile"])
            #'mansard': RoofMansard()
        }
        self.volumeGeneratorMultiFlat = RoofFlatMulti(data, self.itemStore, self.itemFactory, facadeRenderer, itemRenderers["RoofFlatMulti"])