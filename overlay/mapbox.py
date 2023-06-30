"""
This file is a part of Blosm addon for Blender.
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

from . import Overlay, getMapboxAccessToken


class Mapbox(Overlay):
    
    baseUrl = "http://[a,b,c,d].tiles.mapbox.com/v4/%s/{z}/{x}/{y}.png?access_token=%s"
    
    def __init__(self, mapId, maxZoom, addonName):
        super().__init__(
            self.baseUrl % (mapId, getMapboxAccessToken(addonName)),
            maxZoom,
            addonName
        )
        self.mapId = mapId
        if mapId == "mapbox.satellite":
            self.imageExtension = "jpg"
    
    def getOverlaySubDir(self):
        return self.mapId
    
    def removeAccessToken(self, url):
        accessTokenPosition = url.find("?access_token=")
        if accessTokenPosition != -1:
            # Remove the access token to decrease the length of the path.
            # Windows and probably the other OS have a limit for the path in the file system.
            url = url[0:accessTokenPosition]
        return url