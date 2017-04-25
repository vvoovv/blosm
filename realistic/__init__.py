import bpy, bmesh
from util.blender import getBmesh, setBmesh


class OperatorMakeWater(bpy.types.Operator):
    bl_idname = "blosm.make_water"
    bl_label = "Make water"
    bl_description = "Select extent for your area of interest on a geographical map"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'
    
    def invoke(self, context, event):
        obj = context.object
        bm = getBmesh(obj)
        # finally a magic function that does everything
        bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=bm.edges)
        setBmesh(obj, bm)
        return {'FINISHED'}


def register():
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)