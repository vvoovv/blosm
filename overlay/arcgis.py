import os
import bpy
from . import Overlay


def getArcgisAccessToken(addonName):
    prefs = bpy.context.preferences.addons
    if addonName in prefs:
        accessToken = prefs[addonName].preferences.arcgisAccessToken
        if not accessToken:
            raise Exception("An access token for ArcGIS overlays isn't set in the addon preferences")
    else:
        j = os.path.join
        with open(j( j(os.path.realpath(__file__), os.pardir), "arcgis_access_token.txt"), 'r') as file:
            accessToken = file.read()
    return accessToken


class Arcgis(Overlay):
    
    baseUrl = "https://ibasemaps-api.arcgis.com/arcgis/rest/services/%s/MapServer/tile/{z}/{y}/{x}?token=%s"
    
    def __init__(self, mapId, maxZoom, addonName):
        super().__init__(
            self.baseUrl % (mapId, getArcgisAccessToken(addonName)),
            maxZoom,
            addonName
        )
    
    def removeAccessToken(self, url):
        accessTokenPosition = url.find("?token=")
        if accessTokenPosition != -1:
            # Remove the access token to decrease the length of the path.
            # Windows and probably the other OS have a limit for the path in the file system.
            url = url[0:accessTokenPosition]
        return url