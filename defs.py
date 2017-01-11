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

class Keys:
    mode3d = "mode3d"


class Bundles:
    pass


price = 2.85
class App:
    file = "4fea502"
    price = price
    url = "https://gumroad.com/l/blender-osm"
    description = "Buy OSM importer without this popup for just {}$".format(price)
    popupStrings = (
        "Support OSM importer!",
        "The free version is 2D only!",
        "Buy paid version with 3D support",
        "for just {}$".format(price)
    )