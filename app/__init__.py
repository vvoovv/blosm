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

import defs


class App:
    
    def __init__(self):
        self.load()
    
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


def register():
    bpy.utils.register_module(__name__)

def unregister():
    bpy.utils.unregister_module(__name__)