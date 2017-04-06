import os
import bpy
from util.blender import makeActive, appendObjectsFromFile


class AreaRenderer:
    
    #assetPath = "assets/base.blend"
    
    def render(self, app):
        layers = app.layers
        # check we have a layer that defines an area (natural or landus)
        for layer in layers:
            if layer.obj and layer.area: break
        else: return
        
        # the terrain Blender object
        terrain = app.terrain.terrain
        
        # a Python list to store Blender groups for brushes (to be deleted later)
        #groups = []
        
        # setup a DYNAMIC_PAINT modifier for <terrain> with canvas surfaces
        makeActive(terrain)
        m = terrain.modifiers.new("Dynamic Paint", 'DYNAMIC_PAINT')
        bpy.ops.dpaint.type_toggle(type='CANVAS')
        bpy.ops.dpaint.surface_slot_remove()
        
        def prepare_dp_surface(surface, group, output_name_a):
            surface.use_antialiasing = True
            surface.use_drying = False
            surface.use_dry_log = False
            surface.use_dissolve_log = False
            surface.brush_group = group
            surface.output_name_a = output_name_a
            surface.output_name_b = ""
        
        for layer in layers:
            layerId = layer.id
            obj = layer.obj
            if obj and layer.area:
                # create a brush group
                group = bpy.data.groups.new("%s_brush" % layerId)
                # add <obj> to the brush group
                group.objects.link(obj)
                # vertex colors
                bpy.ops.dpaint.surface_slot_add()
                surface = m.canvas_settings.canvas_surfaces.active
                surface.name = "%s_colors" % layerId
                # create a target vertex colors layer for dynamically painted vertex colors
                colors = terrain.data.vertex_colors.new(layerId)
                prepare_dp_surface(surface, group, colors.name)
                # vertex weights
                bpy.ops.dpaint.surface_slot_add()
                surface = m.canvas_settings.canvas_surfaces.active
                surface.name = "%s_weights" % layerId
                surface.surface_type = 'WEIGHT'
                # create a target vertex group for dynamically painted vertex weights
                weights = terrain.vertex_groups.new(layerId)
                prepare_dp_surface(surface, group, weights.name)
        m.canvas_settings.canvas_surfaces.active_index = 0
        m.canvas_settings.canvas_surfaces[0].show_preview = True
        terrain.select = False
        
        # setup a DYNAMIC_PAINT modifier for <layer.obj> in the brush mode
        for layer in layers:
            layerId = layer.id
            obj = layer.obj
            if obj and layer.area:
                makeActive(obj)
                m = obj.modifiers.new("Dynamic Paint", 'DYNAMIC_PAINT')
                m.ui_type = 'BRUSH'
                bpy.ops.dpaint.type_toggle(type='BRUSH')
                brush = m.brush_settings
                brush.paint_color = (1., 1., 1.)
                brush.paint_source = 'DISTANCE'
                brush.paint_distance = 500.
                brush.use_proximity_project = True
                brush.ray_direction = 'Z_AXIS'
                brush.proximity_falloff = 'CONSTANT'
                obj.hide = True
                # deselect <obj> to ensure correct work of subsequent operators
                obj.select = False