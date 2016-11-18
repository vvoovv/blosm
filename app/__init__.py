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