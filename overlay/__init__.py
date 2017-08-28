

class Overlay:
    
    tileWidth = 256
    tileHeight = 256
    
    tileCoordsTemplate = "{z}/{x}/{y}"
    
    subdomains = None
    numSubdomains = 0
    
    tileCounter = 0
    
    def __init__(self, url, addonName):
        # where to stop searching for sundomains {suddomain1,subdomain2}
        subdomainsEnd = len(url)
        # check if have {z}/{x}/{y} in <url> (i.e. tile coords)
        coordsPosition = url.find(self.tileCoordsTemplate)
        if coordsPosition > 0:
            subdomainsEnd = coordsPosition
            urlEnd = url[coordsPosition+len(self.tileCoordsTemplate):]
        else:
            if url[-1] != '/':
                url = url + '/'
            urlEnd = ".png"
        leftBracketPosition = url.find("{", 0, subdomainsEnd)
        rightBracketPosition = url.find("}", leftBracketPosition+2, subdomainsEnd)
        if leftBracketPosition > -1 and rightBracketPosition > -1:
            self.subdomains = tuple(
                s.strip() for s in url[leftBracketPosition+1:rightBracketPosition].split(',')
            )
            self.numSubdomains = len(self.subdomains)
            urlStart = url[:leftBracketPosition]
            urlMid = url[rightBracketPosition+1:coordsPosition]\
                if coordsPosition > 0 else\
                url[rightBracketPosition+1:]
        else:
            urlStart = url[rightBracketPosition+1:coordsPosition] if coordsPosition > 0 else url
            urlMid = None
        self.urlStart = urlStart
        self.urlMid = urlMid
        self.urlEnd = urlEnd
    
    def getTileUrl(self, zoom, x, y):
        if self.subdomains:
            url = "%s%s%s%s/%s/%s%s" % (
                self.urlStart,
                self.subdomains[self.tileCounter % self.numSubdomains],
                self.urlMid,
                zoom,
                x,
                y,
                self.urlEnd
            )
            self.tileCounter += 1
        else:
            url = "%s%s/%s/%s%s" % (
                self.urlStart,
                zoom,
                x,
                y,
                self.urlEnd
            )
        return url


from .mapbox import Mapbox


overlayTypeData = {
    'mapbox-satellite': (Mapbox, "mapbox.satellite"),
    'osm-mapnik': (Overlay, "http://{a,b,c}.tile.openstreetmap.org"),
    'mapbox-streets': (Mapbox, "mapbox.streets"),
    'custom': (Overlay, '')
}