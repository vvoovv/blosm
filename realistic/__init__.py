import bpy, bmesh
from app import app
from util.blender import getBmesh, setBmesh
from .renderer import WaterRenderer


class OperatorMakeWater(bpy.types.Operator):
    bl_idname = "blosm.make_water"
    bl_label = "Make water"
    bl_description = "Select extent for your area of interest on a geographical map"
    bl_options = {'REGISTER', 'UNDO'}
    
    layerId = "water"
    
    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'
    
    def invoke(self, context, event):
        obj = context.object
        bm = getBmesh(obj)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        # a magic function that does everything
        bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=bm.edges)
        setBmesh(obj, bm)
        
        app.setAttributes(context)
        app.setTerrain(context)
        # create a renderer
        renderer = WaterRenderer()
        layer = app.getLayer(self.layerId)
        if not layer:
            layer = app.createLayer(self.layerId)
        layer.obj = obj
        renderer.finalizeBlenderObject(layer, app)
        renderer.renderArea(layer, app)
        return {'FINISHED'}


def register():
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)