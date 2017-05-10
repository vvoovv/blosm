import bpy, bmesh
from app import app
from util.blender import getBmesh, setBmesh
from .renderer import AreaRenderer, ForestRenderer, WaterRenderer


class OperatorMakeRealistic(bpy.types.Operator):
    bl_idname = "blosm.make_realistic"
    bl_label = "Make realistic"
    bl_description = "Make realistic representation on the terrain for the active object with areas"
    bl_options = {'REGISTER', 'UNDO'}
    
    layerId = "water"
    
    @classmethod
    def poll(cls, context):
        return context.scene.objects.get( context.scene.blender_osm.terrainObject )
    
    def invoke(self, context, event):
        obj = context.object
        addon = context.scene.blender_osm
        layerId = addon.makeRealisticLayer
        
        # remove all modifiers
        for i in range(len(obj.modifiers)):
            obj.modifiers.remove(obj.modifiers[i])
        
        # set z-coordinate for all vertices of the input mesh to zero
        bm = getBmesh(obj)
        for v in bm.verts:
            v.co[2] = 0.
        setBmesh(obj, bm)
        
        app.setAttributes(context)
        app.setTerrain(context, False)
        layer = app.getLayer(self.layerId)
        if not layer:
            layer = app.createLayer(self.layerId, swOffset = app.swOffsetDp)
        layer.obj = obj
        
        # Place the input Blender object at the right location in order for
        # the BOOLEAN and SHRINKWRAP modifiers to work correctly
        obj.location = layer.location
        
        # create a renderer
        if layerId == "water":
            renderer = WaterRenderer()
        elif layerId == "forest":
            renderer = ForestRenderer()
        else:
            renderer = AreaRenderer()
            
        renderer.finalizeBlenderObject(layer, app)
        renderer.renderArea(layer, app)
        return {'FINISHED'}


class OperatorMakePolygon(bpy.types.Operator):
    bl_idname = "blosm.make_polygon"
    bl_label = "Make polygon"
    bl_description = "Make a polygon out of connected edges"
    bl_options = {'REGISTER', 'UNDO'}
    
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
        return {'FINISHED'}


def register():
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)