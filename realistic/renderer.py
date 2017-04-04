import os
import bpy
from util.blender import makeActive, appendObjectsFromFile

class AreaRenderer:
    
    assetPath = "assets/base.blend"
    
    def render(self, app):
        # the terrain Blender object
        terrain = app.terrain.terrain
        
        # Append objects with DYNAMIC_PAINT modifiers to be copied to
        # the terrain Blender object, which is to serve as a dynamic paint canvas and
        # to the Blender objects representing flat layers which are to serve
        # as dynamic paint brushes
        dp_brush, dp_canvas = appendObjectsFromFile(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), self.assetPath),
            "dp_brush",
            "dp_canvas"
        )
        
        # Copy the DYNAMIC_PAINT modifier from <dp_canvas> to <terrain>. The modifier turns
        # <terrain> into dynamic paint canvas
        terrain.select = True
        makeActive(dp_canvas)
        bpy.ops.object.make_links_data(type='MODIFIERS')
        terrain.select = False
        # <dp_canvas> isn't needed anymore
        bpy.data.objects.remove(dp_canvas, True)
        
        # DYNAMIC_PAINT modifier of <terrain>
        dp = terrain.modifiers[-1]
        
        # a Python list to store Blender objects used as brushes for dynamic painting
        brushes = []
        # a Python list to store Blender groups for brushes (to be deleted later)
        groups = []
        
        surfaces = dp.canvas_settings.canvas_surfaces
        for layer in app.layers:
            layerId = layer.id
            nameColors = "%s_colors" % layerId
            nameWeights = "%s_weights" % layerId
            obj = layer.obj
            if obj and nameColors in surfaces and nameWeights in surfaces:
                obj.select = True
                brushes.append(obj)
                surfaceColors = surfaces[nameColors]
                surfaceWeights = surfaces[nameWeights]
                surfaceColors.is_active = True
                surfaceWeights.is_active = True
                # create a brush group
                group = bpy.data.groups.new("%s_brush" % layerId)
                # add <obj> to the brush group
                group.objects.link(obj)
                surfaceColors.brush_group = group
                surfaceWeights.brush_group = group
                # create a target vertex colors layer for dynamically painted vertex colors
                colors = terrain.data.vertex_colors.new(layerId)
                surfaceColors.output_name_a = colors.name
                # create a target vertex group for dynamically painted vertex weights
                weights = terrain.vertex_groups.new(layerId)
                surfaceWeights.output_name_a = weights.name
            # Copy the DYNAMIC_PAINT modifier from <dp_brush> to the Blender objects from 
            # the Python list <brushes>. The modifier turns them into dynamic paint brushes
            makeActive(dp_brush)
            bpy.ops.object.make_links_data(type='MODIFIERS')
            for brush in brushes:
                brush.select = False
            # <dp_brush> isn't needed anymore
            #bpy.data.objects.remove(dp_brush, True)