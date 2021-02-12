import os
from urllib import request


class BaseApp:
    
    osmFileName = "map%s.osm"
    
    osmUrlPath = "/api/map?bbox=%s,%s,%s,%s"
    
    def download(self, url, filepath, data=None):
        print("Downloading the file from %s..." % url)
        if data:
            data = data.encode('ascii')
        request.urlretrieve(url, filepath, None, data)
        print("Saving the file to %s..." % filepath)
    
    def downloadOsmFile(self, osmDir, minLon, minLat, maxLon, maxLat):
        # find a file name for the OSM file
        osmFileName = self.osmFileName % ""
        counter = 1
        while True:
            osmFilepath = os.path.realpath( os.path.join(osmDir, osmFileName) )
            if os.path.isfile(osmFilepath):
                counter += 1
                osmFileName = self.osmFileName % "_%s" % counter
            else:
                break
        self.osmFilepath = osmFilepath
        self.download(
            self.osmServer + self.osmUrlPath % (minLon, minLat, maxLon, maxLat),
            osmFilepath
        )