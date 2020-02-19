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

from parse.osm.relation.building import Building

from building.manager import BuildingParts, BuildingRelations

from manager.logging import Logger

from realistic.building.manager import RealisticBuildingManager
from realistic.building.renderer import RealisticBuildingRenderer


def setup_base(app, osm, getMaterials, bldgPreRender):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    if app.buildings:
        buildingParts = BuildingParts()
        buildingRelations = BuildingRelations()
        buildings = RealisticBuildingManager(osm, buildingParts)
        
        # Important: <buildingRelation> beform <building>,
        # since there may be a tag building=* in an OSM relation of the type 'building'
        osm.addCondition(
            lambda tags, e: isinstance(e, Building),
            None,
            buildingRelations
        )
        osm.addCondition(
            lambda tags, e: "building" in tags,
            "buildings",
            buildings
        )
        osm.addCondition(
            lambda tags, e: "building:part" in tags,
            None,
            buildingParts
        )
        # set building renderer
        br = RealisticBuildingRenderer(
            app,
            bldgPreRender = bldgPreRender,
            materials = getMaterials()
        )
        # <br> stands for "building renderer"
        buildings.setRenderer(br)
        app.managers.append(buildings)
    
    if app.forests:
        setup_forests(app, osm)


def setup_forests(app, osm):
    from renderer import Renderer2d
    from realistic.manager import AreaManager
    from realistic.renderer import AreaRenderer, ForestRenderer
    from renderer.node_renderer import SingleTreeRenderer
    
    areaRenderers = dict(forest=ForestRenderer())
    # create managers
    m = AreaManager(osm, app, AreaRenderer(), **areaRenderers)
    
    osm.addCondition(
        lambda tags, e: tags.get("natural") == "wood" or tags.get("landuse") == "forest",
        "forest",
        m
    )
    
    osm.addNodeCondition(
        lambda tags, e: tags.get("natural") == "tree",
        "trees",
        None,
        SingleTreeRenderer(app)
    )
    
    m.setRenderer(Renderer2d(app, applyMaterial=False))
    app.managers.append(m)