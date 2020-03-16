"""
This file is part of blender-osm (OpenStreetMap importer for Blender).
Copyright (C) 2014-2018 Vladimir Elistratov
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

import bpy
from manager import BaseManager
from renderer import Renderer
from .renderer import AreaRenderer#, TerrainRenderer

_isBlender280 = bpy.app.version[1] >= 80


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
        #self.terrainRenderer = TerrainRenderer()

    def createLayer(self, layerId, app, **kwargs):
        if not app.singleObject:
            # patching kwargs: forcing single object mode
            kwargs["singleObject"] = True
        return super().createLayer(layerId, app, **kwargs)

    def parseWay(self, element, elementId):
        # exactly the same code as in manager.Polygon.parseWay(..)
        if element.closed:
            element.t = Renderer.polygon
            # render it in <BaseManager.render(..)>
            element.r = True
        else:
            element.valid = False
    
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
            if layer.area and layer.obj and layer.renderer:
                layer.renderer.finalizeBlenderObject(layer, app)
        
        AreaRenderer.addSubsurfModifier(terrain)
        AreaRenderer.beginDynamicPaintCanvas(terrain)
                
        for layer in app.layers:
            if layer.area and layer.obj and layer.renderer:
                layer.renderer.renderTerrain(layer, terrain)
        
        AreaRenderer.endDynamicPaintCanvas(terrain)
        
        for layer in app.layers:
            if layer.area and layer.obj and layer.renderer:
                layer.renderer.renderArea(layer, app)
        
        # set material for the terrain
        #self.terrainRenderer.render(app)
        
        if _isBlender280:
            terrain.select_set(False)
        else:
            terrain.select = False