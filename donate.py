import bpy
import webbrowser

url = "https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=NNQBWQ6TH2N7N"


class Donate(bpy.types.Operator):
    bl_idname = "blender_geo.donate"
    bl_label = "Donate!"
    bl_description = "If you like the add-on please donate"
    bl_options = {"REGISTER"}
    
    def execute(self, context):
        webbrowser.open_new_tab(url)
        return {'FINISHED'}
    
    @classmethod
    def gui(cls, layout, addonName):
        box = layout.box()
        box.label("If you like \'{}\' add-on".format(addonName))
        box.label("please donate!")
        
        box.operator(cls.bl_idname, icon='HELP')