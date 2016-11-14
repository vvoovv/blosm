import bpy
import os, webbrowser

licenseKey = "4fea502"
price = 2.85
url = "https://gumroad.com/l/blender-osm"

def hasLicenseFile():
    # app directory
    directory = os.path.dirname(os.path.realpath(__file__))
    # app/..
    directory = os.path.realpath( os.path.join(directory, os.pardir) )
    return os.path.isfile( os.path.join(directory, licenseKey) )


class OperatorPopup(bpy.types.Operator):
    bl_idname = "prk.check_version_osm"
    bl_label = ""
    bl_description = "Buy OSM importer without this popup for just {}$".format(price)
    bl_options = {'INTERNAL'}
    
    def invoke(self, context, event):
        if hasLicenseFile():
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)
    
    def modal(self, context, event):
        print("event")
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        webbrowser.open_new_tab(url)
        return {'FINISHED'}
    
    def cancel(self, context):
        webbrowser.open_new_tab(url)
    
    def draw(self, context):
        layout = self.layout
        
        self.label("Support OSM importer!", icon='INFO')
        self.label("Buy OSM importer without this popup")
        self.label("for just {}$".format(price))
        
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