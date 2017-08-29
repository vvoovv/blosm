import bpy
from . import Overlay


class Mapbox(Overlay):
    
    subdomains = ("a", "b", "c", "d")
    
    baseUrl = "http://{a,b,c,d}.tiles.mapbox.com/v4/%s/{z}/{x}/{y}.png?access_token=%s"
    
    def __init__(self, mapId, maxZoom, addonName):
        super().__init__(
            self.baseUrl % (mapId, Mapbox.getAccessToken(addonName)),
            maxZoom,
            addonName
        )
        self.mapId = mapId
    
    def getOverlaySubDir(self):
        return self.mapId
    
    @staticmethod
    def getAccessToken(addonName):
        prefs = bpy.context.user_preferences.addons
        if addonName in prefs:
            accessToken = prefs[addonName].preferences.mapboxAccessToken
            if not accessToken:
                raise Exception("An access token for Mapbox overlays isn't set in the addon preferences")
        else:
            import os
            j = os.path.join
            with open(j( j(os.path.realpath(__file__), os.pardir), "access_token.txt"), 'r') as file:
                accessToken = file.read()
        return accessToken