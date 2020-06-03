import xml.etree.cElementTree as etree


class Gpx:
    """
    Representation of data in a GPX file
    """
    
    def __init__(self, app):
        self.app = app
        
        self.projection = None
        
        # a list of track segments (trkseg)
        self.segments = []
    
    def parse(self, filepath):
        
        projection = self.projection
        if not self.projection:
            self.minLat = 90.
            self.maxLat = -90.
            self.minLon = 180.
            self.maxLon = -180.
        
        gpx = etree.parse(filepath).getroot()
        
        for e1 in gpx: # e stands for element
            # Each tag may have the form {http://www.topografix.com/GPX/1/1}tag
            # That's whay we skip curly brackets
            if e1.tag[e1.tag.find("}")+1:] == "trk":
                for e2 in e1:
                    if e2.tag[e2.tag.find("}")+1:] == "trkseg":
                        segment = []
                        for e3 in e2:
                            if e3.tag[e3.tag.find("}")+1:] == "trkpt":
                                lat = float(e3.attrib["lat"])
                                lon = float(e3.attrib["lon"])
                                
                                if not projection:
                                    self.updateBounds(lat, lon)
                                # check if <trkpt> has <ele>
                                ele = None
                                for e4 in e3:
                                    if e4.tag[e4.tag.find("}")+1:] == "ele":
                                        ele = e4
                                        break
                                point = (lat, lon, float(ele.text)) if not ele is None else (lat, lon, 0.)
                                segment.append(point)
                        self.segments.append(segment)

        if not projection:
            # set projection using the calculated bounds (self.minLat, self.maxLat, self.minLon, self.maxLon)
            self.setProjection(
                (self.minLat + self.maxLat)/2.,
                (self.minLon + self.maxLon)/2.
            )
    
    def updateBounds(self, lat, lon):
        if lat < self.minLat:
            self.minLat = lat
        elif lat > self.maxLat:
            self.maxLat = lat
        if lon < self.minLon:
            self.minLon = lon
        elif lon > self.maxLon:
            self.maxLon = lon
    
    def setProjection(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.app.setProjection(lat, lon)