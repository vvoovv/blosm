import os, math

from grammar import *
from grammar.scope import PerBuilding, PerFootprint
from grammar import units, symmetry, smoothness
from grammar.value import Value, FromAttr, FromBldgAttr, Alternatives, Conditional, FromStyleBlockAttr, Constant
from grammar.value import RandomWeighted, RandomNormal
from action.volume.roof import Roof as RoofDefs
from item.defs import *

from pml import PML


minHeightForLevels = 1.5
minWidthForOpenings = 1.


class StyleStore:
    
    def __init__(self, app, styles=None):
        self.styles = {}
        # overwrite an entry with the given key in <self.styles> if the key already exists in <self.styles>
        self.overwrite = True
        
        if styles:
            self.addStyles(styles)
        elif app.pmlFilepath:
            self.loadFromFile(app.pmlFilepath, app.assetsDir)
    
    def addStyles(self, styles):
        for styleName in styles:
            self.add(styleName, styles[styleName]) 
    
    def add(self, styleName, style):
        style = Grammar(style)
        self.styles[styleName] = style
    
    def get(self, styleName):
        return self.styles[styleName]
    
    def loadFromFiles(self, files):
        for file in files:
            if os.path.isfile(file) and file.lower().endswith(".pml"):
                self.loadFromFile(file)
    
    def loadFromFile(self, file, assetsDir):
        _locals = {}
        exec(PML(file, assetsDir).getPythonCode(), None, _locals)
        styles = _locals["styles"]
        if isinstance(styles, dict):
            self.addStyles(styles)
        else: # a Python list
            # use the file name without the extension as the style name
            styleName = os.path.splitext(os.path.basename(file))[0]
            self.add(styleName, styles)