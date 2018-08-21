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

from setup.premium import setup_base

def setup(app, osm):
    setup_base(app, osm, getMaterials, bldgPreRender)


from realistic.material.renderer import UvOnly


def getMaterials():
    return dict(
        uv_only = UvOnly
    )


def bldgPreRender(building, app):
    building.setMaterialWalls("uv_only")
    building.setMaterialRoof("uv_only")