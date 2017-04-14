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

from manager import BaseManager
from .renderer import AreaRenderer
from util.blender import makeActive


class AreaManager(BaseManager):
    
    def __init__(self, osm, app, defaultRenderer, **renderers):
        """
        Args:
            osm (parse.Osm): Parsed OSM data
        """
        super().__init__(osm)
        self.app = app
        self.defaultRenderer = defaultRenderer
        self.renderers = renderers
    
    def renderExtra(self):
        app = self.app
        # Blender object for the terrain
        terrain = app.terrain.terrain
        
        # a counter for the number of valid area layers
        numLayers = 0
        # set renderer for each layer
        for layer in app.layers:
            layerId = layer.id
            if layer.area and layer.obj:
                numLayers += 1
                # find a render for <layer>
                for _layerId in self.renderers:
                    if _layerId.startswith(layerId):
                        renderer = self.renderers[layerId]
                        break
                else:
                    renderer = self.defaultRenderer
                # save <renderer> in <layer> for future references
                layer.renderer = renderer
        if not numLayers:
            return
        
        for layer in app.layers:
            if layer.area and layer.obj:
                layer.renderer.finalizeBlenderObject(layer, app)
        
        AreaRenderer.addSubsurfModifier(terrain)
        AreaRenderer.beginDynamicPaintCanvas(terrain)
                
        for layer in app.layers:
            if layer.area and layer.obj:
                layer.renderer.renderTerrain(layer, terrain)
        
        AreaRenderer.endDynamicPaintCanvas(terrain)
        
        for layer in app.layers:
            if layer.area and layer.obj:
                layer.renderer.renderArea(layer, app)
        
        terrain.select = False