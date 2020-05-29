import os, math

from grammar import *
from grammar.scope import PerBuilding, PerFootprint
from grammar import units, symmetry, smoothness
from grammar.value import Value, FromAttr, Alternatives, Conditional, FromStyleBlockAttr, Constant
from grammar.value import RandomWeighted, RandomNormal
from action.volume.roof import Roof as RoofDefs
from item.defs import RoofShapes, CladdingMaterials


minHeightForLevels = 1.5
minWidthForOpenings = 1.


class StyleStore:
    
    def __init__(self, styles):
        self.styles = {}
        # overwrite an entry with the given key in <self.styles> if the key already exists in <self.styles>
        self.overwrite = True
        
        if styles:
            for styleName in styles:
                self.add(styleName, styles[styleName])
        else:
            pass
    
    def add(self, styleName, style):
        style = Grammar(style)
        self.styles[styleName] = style
    
    def get(self, styleName):
        return self.styles[styleName]
    
    def loadFromDirectory(self, directory):
        if not os.path.isdir(directory):
            raise Exception(
                "The directory with PML files %s doesn't exist. " % directory
            )
        for file in os.listdir(directory):
            if os.path.isfile(file) and file.lower().endswith(".pml"):
                self.loadFromFile(file)
    
    def loadFromFiles(self, files):
        for file in files:
            if os.path.isfile(file) and file.lower().endswith(".pml"):
                self.loadFromFile(file)
    
    def loadFromFile(self, file):
        pass