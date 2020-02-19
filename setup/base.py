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

from manager import BaseManager, Linestring, Polygon, PolygonAcceptBroken, WayManager
from renderer import Renderer2d
from renderer.node_renderer import BaseNodeRenderer
from renderer.curve_renderer import CurveRenderer

from building.manager import BuildingManager, BuildingParts, BuildingRelations
from building.renderer import BuildingRenderer

from manager.logging import Logger


def tunnel(tags, e):
    if tags.get("tunnel") == "yes":
        e.valid = False
        return True
    return False


def setup(app, osm):
    # comment the next line if logging isn't needed
    Logger(app, osm)
    
    # create managers
    wayManager = WayManager(osm, CurveRenderer(app))
    linestring = Linestring(osm)
    polygon = Polygon(osm)
    polygonAcceptBroken = PolygonAcceptBroken(osm)
    
    # conditions for point objects in OSM
    #osm.addNodeCondition(
    #    lambda tags, e: tags.get("natural") == "tree",
    #    "trees",
    #    None,
    #    BaseNodeRenderer(app, path, filename, collection)
    #)
    
    if app.buildings:
        if app.mode is app.twoD:
            osm.addCondition(
                lambda tags, e: "building" in tags,
                "buildings", 
                polygon
            )
        else: # 3D
            buildingParts = BuildingParts()
            buildingRelations = BuildingRelations()
            buildings = BuildingManager(osm, buildingParts)
            
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
            buildings.setRenderer(
                BuildingRenderer(app)
            )
            app.managers.append(buildings)
    
    if app.highways or app.railways:
        osm.addCondition(tunnel)
    
    if app.highways:
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("motorway", "motorway_link"),
            "roads_motorway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("trunk", "trunk_link"),
            "roads_trunk",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("primary", "primary_link"),
            "roads_primary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("secondary", "secondary_link"),
            "roads_secondary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("tertiary", "tertiary_link"),
            "roads_tertiary",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "unclassified",
            "roads_unclassified",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("residential", "living_street"),
            "roads_residential",
            wayManager
        )
        # footway to optimize the walk through conditions
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("footway", "path"),
            "paths_footway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "service",
            "roads_service",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "pedestrian",
            "roads_pedestrian",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "track",
            "roads_track",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "steps",
            "paths_steps",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "cycleway",
            "paths_cycleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") == "bridleway",
            "paths_bridleway",
            wayManager
        )
        osm.addCondition(
            lambda tags, e: tags.get("highway") in ("road", "escape", "raceway"),
            "roads_other",
            wayManager
        )
    if app.railways:
        osm.addCondition(
            lambda tags, e: "railway" in tags,
            "railways",
            wayManager
        )
    if app.water:
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "water" or tags.get("waterway") == "riverbank" or tags.get("landuse") == "reservoir",
            "water",
            polygonAcceptBroken
        )
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "coastline",
            "coastlines",
            linestring
        )
    if app.forests:
        osm.addCondition(
            lambda tags, e: tags.get("natural") == "wood" or tags.get("landuse") == "forest",
            "forest",
            polygon
        )
    if app.vegetation:
        osm.addCondition(
            lambda tags, e: ("landuse" in tags and tags["landuse"] in ("grass", "meadow", "farmland")) or ("natural" in tags and tags["natural"] in ("scrub", "grassland", "heath")),
            "vegetation",
            polygon
        )
    
    numConditions = len(osm.conditions)
    if not app.mode is app.twoD and app.buildings:
        # 3D buildings aren't processed by BaseManager
        numConditions -= 1
    if numConditions:
        m = BaseManager(osm)
        m.setRenderer(Renderer2d(app))
        app.managers.append(m)