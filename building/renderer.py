from renderer import Renderer, Renderer3d
from manager import Manager

# Python tuples to store some defaults to render walls and roofs of OSM 3D buildings
# Indices to access defaults from Python tuple below
roofIndex = 0
wallIndex = 1
tags = ("roof:colour", "building:colour")
defaultColorNames = ("roof", "wall")
defaultColors = ( (0.309, 0.013, 0.012), (1., 0.5, 0.2) )


class BuildingRenderer(Renderer3d):
    
    def __init__(self, op, layerId):
        super().__init__(op)
        self.layerIndex = op.layerIndices.get(layerId)
    
    def render(self, building, osm):
        parts = building.parts
        outline = building.element
        self.parts = parts
        self.outline = outline
        self.preRender(outline, self.layerIndex)
        
        if parts:
            # reset material indices derived from <outline>
            self.defaultMaterialIndices = [None, None]
        if not parts or outline.tags.get("building:part") == "yes":
            # render building outline
            self.renderElement(outline, building, osm)
        if parts:
            for part in parts:
                self.renderElement(part, building, osm)
        self.postRender(outline)
    
    def renderElement(self, element, building, osm):
        # set attributes used be the parent class <Renderer3d>
        self.z = building.getHeight(element, self.op)
        self.h = self.z - building.getMinHeight(element, self.op)
        
        if element.t is Renderer.polygon:
            self.renderPolygon(element, osm)
        else:
            self.renderMultiPolygon(element, osm)

    def getMaterialIndex(self, element):
        """
        Returns the material index for the building roof
        
        Args:
            element: OSM element (building=* or building:part=*)
        """
        return self.getMaterialIndexByPart(element, roofIndex)
    
    def getMaterialIndexByPart(self, element, partIndex):
        """
        Returns the material index either for building walls or for buildings roofs
        
        Args:
            element: OSM element (building=* or building:part=*)
            partIndex (int): Equal to either <roofIndex> or <wallIndex>
        """
        # material name is just the related color (either a hex or a CSS color)
        name = Manager.normalizeColor(element.tags.get(tags[partIndex]))
        
        if name is None:
            # <name> is invalid as a color name
            if self.outline is element:
                # Building oultine is rendererd if there are no parts or
                # if the outline has <building:part=yes>
                
                # take the name for the related default color
                name = defaultColorNames[partIndex]
                # check if Blender material has been already created
                materialIndex = self.getMaterialIndexByName(name)
                if materialIndex is None:
                    # the related Blender material hasn't been created yet, so create it
                    materialIndex = self.createDiffuseMaterial(name, defaultColors[partIndex])
                if self.parts:
                    # If there are parts, store <materialIndex>,
                    # since it's set for a building part, if it doesn't have an OSM tag for color
                    self.defaultMaterialIndices[partIndex] = materialIndex
            else:
                # this a building part (building:part=yes), not the building outline
                
                # check if the material index for the default color has been set before
                if self.defaultMaterialIndices[partIndex] is None:
                    # The material index for the default color hasn't been set before
                    # Get the material index for the default color,
                    # i.e. the material index for <self.outline>
                    materialIndex = self.getMaterialIndexByPart(self.outline, partIndex)
                else:
                    # The material index for the default color has been set before, so use it,
                    # i.e. use the color for <self.outline>
                    materialIndex = self.defaultMaterialIndices[partIndex]
        else:
            # check if the related Blender material has been already created
            materialIndex = self.getMaterialIndexByName(name)
            if materialIndex is None:
                # The related Blender material hasn't been already created,
                # so create it
                materialIndex = self.createDiffuseMaterial(name, Manager.getColor(name))
            # If <element is self.outline> and there are parts, store <materialIndex>,
            # since it's set for a building part, if it doesn't have an OSM tag for color
            if element is self.outline and self.parts:
                self.defaultMaterialIndices[partIndex] = materialIndex
        return materialIndex
    
    def getSideMaterialIndex(self, element):
        """
        Returns the material index for the building walls
        
        Args:
            element: OSM element (building=* or building:part=*)
        """
        return self.getMaterialIndexByPart(element, wallIndex)
        