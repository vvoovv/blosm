from . import Overlay


class Arcgis(Overlay):
    
    baseUrl = "https://ibasemaps-api.arcgis.com/arcgis/rest/services/%s/MapServer/tile/{z}/{y}/{x}?token=%s"
    
    def __init__(self, mapId, maxZoom, app):
        super().__init__(
            self.baseUrl % (mapId, app.getArcgisAccessToken()),
            maxZoom,
            app
        )
        if mapId == "World_Imagery":
            self.imageExtension = "jpg"
    
    def removeAccessToken(self, url):
        accessTokenPosition = url.find("?token=")
        if accessTokenPosition != -1:
            # Remove the access token to decrease the length of the path.
            # Windows and probably the other OS have a limit for the path in the file system.
            url = url[0:accessTokenPosition]
        return url