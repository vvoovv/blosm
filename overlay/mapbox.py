from . import Overlay


class Mapbox(Overlay):
    
    subdomains = ("a", "b", "c", "d")
    
    baseUrl = "http://{a,b,c,d}.tiles.mapbox.com/v4/%s/{z}/{x}/{y}.png?access_token=%s"
    
    def __init__(self, mapId):
        super().__init__(self.baseUrl % (mapId, Mapbox.getAccessToken()))
    
    @staticmethod
    def getAccessToken():
        return "zxcvb"