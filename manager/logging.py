from datetime import datetime
from building.manager import BuildingManager


class Logger:
    
    def __init__(self, op, osm):
        self.parseStartTime = datetime.now()
        op.logger = self
        self.op = op
        self.osm = osm
    
    def processStart(self):
        print("Parsing OSM file: {}".format(datetime.now() - self.parseStartTime))
        self.processStartTime = datetime.now()

    def processEnd(self):
        self.numBuildings()
        print("Processing of parsed OSM data: {}".format(datetime.now() - self.processStartTime))
    
    def renderStart(self):
        self.renderStartTime = datetime.now()

    def renderEnd(self):
        t = datetime.now()
        print("Mesh creation in Blender: {}".format(t - self.renderStartTime))
        print("Total duration: {}".format(t - self.parseStartTime))
    
    def numBuildings(self):
        op = self.op
        if not (op.mode == '3D' and op.buildings):
            return
        for m in op.managers:
            if isinstance(m, BuildingManager):
                print("The number of buildings: {}".format(len(m.buildings)))