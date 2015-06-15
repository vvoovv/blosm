bl_info = {
    "name": "One-vertex mesh object at the cursor location",
    "author": "Vladimir Elistratov <vladimir.elistratov@gmail.com>",
    "version": (1, 0, 0),
    "blender": (2, 6, 9),
    "location": "View 3D > Object Mode > Tool Shelf",
    "description": "Create one-vertex mesh object at the cursor location",
    "warning": "",
    "wiki_url": "https://github.com/vvoovv/blender-geo/wiki/One-vertex-mesh-object-at-the-cursor-location",
    "tracker_url": "https://github.com/vvoovv/blender-geo/issues",
    "support": "COMMUNITY",
    "category": "3D View",
}

import bpy

class PlaceVertexAtCursorPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_label = "New Object"

    def draw(self, context):
        c = self.layout.column()
        c.operator("object.vertex_object_at_cursor")

class PlaceVertexAtCursor(bpy.types.Operator):
    bl_idname = "object.vertex_object_at_cursor"
    bl_label = "One-vertex object at the cursor"
    bl_options = {"UNDO"}

    bl_description = "Create one-vertex mesh object at the cursor location"
    
    def execute(self, context):
        # setting active object if there is no active object
        if not context.scene.objects.active:
            context.scene.objects.active = context.scene.objects[0]
        bpy.ops.object.mode_set(mode="OBJECT")
        
        mesh = bpy.data.meshes.new("")
        mesh.from_pydata([context.scene.cursor_location], [], [])
        mesh.update()
        obj = bpy.data.objects.new("", mesh)
        context.scene.objects.link(obj)
        bpy.ops.object.select_all(action = "DESELECT")
        obj.select = True
        context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}

def register():
    bpy.utils.register_module(__name__)

def unregister():
    bpy.utils.unregister_module(__name__)