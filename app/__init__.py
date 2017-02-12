"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2017 Vladimir Elistratov
prokitektura+support@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import bpy
import os, json, webbrowser, base64
from urllib import request

import defs
from renderer import Renderer


class App:
    
    layers = ("buildings", "highways", "railways", "water", "forests", "vegetation")
    
    osmUrl = "http://overpass-api.de/api/map"
    
    osmDir = "osm"
    
    osmFileName = "map%s.osm"
    
    # diffuse colors for some layers
    colors = {
        "buildings": (0.309, 0.013, 0.012),
        "highways": (0.1, 0.1, 0.1),
        "water": (0.009, 0.002, 0.8),
        "forests": (0.02, 0.208, 0.007),
        "vegetation": (0.007, 0.558, 0.005),
        "railways": (0.2, 0.2, 0.2)
    }
    
    def __init__(self):
        self.load()
    
    def init(self, op, context, basePath, addonName):
        self.op = op
        self.assetPath = os.path.join(basePath, "assets")
        # set <self.dataDir> (path to data)
        prefs = context.user_preferences.addons
        j = os.path.join
        if addonName in prefs:
            dataDir = prefs[addonName].preferences.dataDir
            if not dataDir:
                raise Exception("A valid directory for data in the addon preferences isn't set")
            dataDir = os.path.realpath(bpy.path.abspath(dataDir))
            if not os.path.isdir(dataDir):
                raise Exception("The directory for data in the addon preferences doesn't exist")
            self.dataDir = dataDir
        else:
            # set <self.dataDir> to basePath/../../../data (development version)
            self.dataDir = os.path.realpath( j( j( j( j(basePath, os.pardir), os.pardir), os.pardir), "data") )
        # create a sub-directory under <self.dataDir> for OSM files
        osmDir = os.path.join(self.dataDir, self.osmDir)
        if not os.path.exists(osmDir):
            os.makedirs(osmDir)
        
        # <self.logger> may be set in <setup(..)>
        self.logger = None
        # a Python dict to cache Blender meshes loaded from Blender files serving as an asset library
        self.meshes = {}
        
        # copy properties from <context.scene.blender_osm>
        addon = context.scene.blender_osm
        for p in dir(addon):
            if not (p.startswith("__") or p in ("bl_rna", "rna_type")):
                setattr(self, p, getattr(addon, p))
        
        if addon.osmSource == "server":
            # find a file name for the OSM file
            osmFileName = self.osmFileName % ""
            counter = 1
            while True:
                osmFilePath = os.path.realpath( os.path.join(osmDir, osmFileName) )
                if os.path.exists(osmFilePath):
                    counter += 1
                    osmFileName = self.osmFileName % "_%s" % counter
                else:
                    break
            self.osmFilepath = self.download(
                "%s?bbox=%s,%s,%s,%s"  % (self.osmUrl, app.minLon, app.minLat, app.maxLon, app.maxLat),
                osmFilePath
            )
        else:
            self.osmFilepath = os.path.realpath(bpy.path.abspath(self.osmFilepath))
        
        if not app.has(defs.Keys.mode3d):
            self.mode = '2D'
        
        # manager (derived from manager.Manager) performing some processing
        self.managers = []
        
        self.prepareLayers()
        if not len(self.layerIndices):
            self.layered = False


    def prepareLayers(self):
        layerIndices = {}
        layerIds = []
        i = 0
        for layerId in self.layers:
            if getattr(self, layerId):
                layerIndices[layerId] = i
                layerIds.append(layerId)
                i += 1
        self.layerIndices = layerIndices
        self.layerIds = layerIds
    
    def process(self):
        logger = self.logger
        if logger: logger.processStart()
        
        for m in self.managers:
            m.process()
        
        if logger: logger.processEnd()
    
    def render(self):
        logger = self.logger
        if logger: logger.renderStart()
        
        Renderer.begin(self)
        for m in self.managers:
            m.render()
        Renderer.end(self)
        
        if logger: logger.renderEnd()
    
    def clean(self):
        self.meshes = None
        self.managers = None
    
    def has(self, key):
        return self.license and (self.all or key in self.keys)
    
    def load(self):
        # this directory
        directory = os.path.dirname(os.path.realpath(__file__))
        # app/..
        directory = os.path.realpath( os.path.join(directory, os.pardir) )
        path = os.path.join(directory, defs.App.file)
        self.license = os.path.isfile(path)
        if not self.license:
            return
        
        with open(path, "r", encoding="ascii") as data:
            data = json.loads( base64.b64decode( bytes.fromhex(data.read()) ).decode('ascii') )
        
        self.all = data.get("all", False)
        self.keys = set(data.get("keys", ()))
    
    def show(self):
        bpy.ops.prk.check_version_osm('INVOKE_DEFAULT')
    
    def download(self, url, filepath):
        print("Downloading the file from %s..." % url)
        request.urlretrieve(url, filepath)
        print("Saving the file to %s..." % filepath)
        return filepath


class OperatorPopup(bpy.types.Operator):
    bl_idname = "prk.check_version_osm"
    bl_label = ""
    bl_description = defs.App.description
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        webbrowser.open_new_tab(defs.App.url)
        return {'FINISHED'}
    
    def cancel(self, context):
        webbrowser.open_new_tab(defs.App.url)
    
    def draw(self, context):
        layout = self.layout
        
        iconPlaced = False
        for label in defs.App.popupStrings:
            if iconPlaced:
                self.label(label)
            else:
                self.label(label, icon='INFO')
                iconPlaced = True
        
        layout.separator()
        layout.separator()
        
        self.label("Click to buy")
    
    def label(self, text, **kwargs):
        row = self.layout.row()
        row.alignment = "CENTER"
        row.label(text, **kwargs)


app = App()


def register():
    bpy.utils.register_module(__name__)

def unregister():
    bpy.utils.unregister_module(__name__)