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

from geojson import Manager, BuildingManager
from renderer import Renderer2d
from renderer.node_renderer import BaseNodeRenderer
from renderer.curve_renderer import CurveRenderer
from building.renderer import BuildingRenderer

from manager.logging import Logger


def tunnel(tags, e):
    if tags.get("tunnel") == "yes":
        e.valid = False
        return True
    return False


def setup(app, data):
    # comment the next line if logging isn't needed
    Logger(app, data)
    
    # create managers
    #wayManager = WayManager(osm, CurveRenderer(app))
    #linestring = Linestring(osm)
    #polygon = Polygon(osm)
    #polygonAcceptBroken = PolygonAcceptBroken(osm)
    manager = Manager(data)
    
    # conditions for point objects in OSM
    #osm.addNodeCondition(
    #    lambda tags, e: tags.get("natural") == "tree",
    #    "trees",
    #    None,
    #    BaseNodeRenderer(app, path, filename, collection)
    #)
    
    if app.buildings:
        if app.mode is app.twoD:
            data.addCondition(
                lambda tags, e: tags.get("building"),
                "buildings", 
                manager
            )
        else: # 3D
            # no building parts for the moment
            buildings = BuildingManager(data, None)
            
            data.addCondition(
                lambda tags, e: tags.get("building"),
                "buildings",
                buildings
            )
            #osm.addCondition(
            #    lambda tags, e: "building:part" in tags,
            #    None,
            #    buildingParts
            #)
            buildings.setRenderer(
                BuildingRenderer(app)
            )
            app.managers.append(buildings)
    
    if app.highways or app.railways:
        data.addCondition(tunnel)
    
    if app.highways:
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("motorway", "motorway_link"),
            "roads_motorway",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("trunk", "trunk_link"),
            "roads_trunk",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("primary", "primary_link"),
            "roads_primary",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("secondary", "secondary_link"),
            "roads_secondary",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("tertiary", "tertiary_link"),
            "roads_tertiary",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "unclassified",
            "roads_unclassified",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("residential", "living_street"),
            "roads_residential",
            manager
        )
        # footway to optimize the walk through conditions
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("footway", "path"),
            "paths_footway",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "service",
            "roads_service",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "pedestrian",
            "roads_pedestrian",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "track",
            "roads_track",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "steps",
            "paths_steps",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "cycleway",
            "paths_cycleway",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") == "bridleway",
            "paths_bridleway",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("highway") in ("road", "escape", "raceway"),
            "roads_other",
            manager
        )
    if app.railways:
        data.addCondition(
            lambda tags, e: "railway" in tags,
            "railways",
            manager
        )
    if app.water:
        data.addCondition(
            lambda tags, e: tags.get("natural") == "water" or tags.get("waterway") == "riverbank" or tags.get("landuse") == "reservoir",
            "water",
            manager
        )
        data.addCondition(
            lambda tags, e: tags.get("natural") == "coastline",
            "coastlines",
            manager
        )
    if app.forests:
        data.addCondition(
            lambda tags, e: tags.get("natural") == "wood" or tags.get("landuse") == "forest",
            "forest",
            manager
        )
    if app.vegetation:
        data.addCondition(
            lambda tags, e: ("landuse" in tags and tags["landuse"] in ("grass", "meadow", "farmland")) or ("natural" in tags and tags["natural"] in ("scrub", "grassland", "heath")),
            "vegetation",
            manager
        )
    
    numConditions = len(data.conditions)
    if not app.mode is app.twoD and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        manager.setRenderer(Renderer2d(app))
        app.managers.append(manager)