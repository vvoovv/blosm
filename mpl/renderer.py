from . import Mpl


class WayRenderer:
    
    def __init__(self):
        self.mpl = Mpl.getMpl()
    
    def render(self, way, data):
        prevCoord = None
        for coord in way.element.getData(data):
            if prevCoord:
                self.mpl.ax.plot(
                    (prevCoord[0], coord[0]),
                    (prevCoord[1], coord[1]),
                    linewidth = 1.,
                    color = "brown"
                )
            prevCoord = coord
    
    def prepare(self):
        pass
    
    def finalize(self):
        self.mpl.show()
    
    def cleanup(self):
        self.mpl = None
        Mpl.cleanup()