import matplotlib.pyplot as plt


class Mpl:
    """
    A wrapper for matplotlib
    """
    mpl = None
    
    def __init__(self):
        self.shown = False
        fig = plt.figure()
        self.ax = fig.gca()
    
    def show(self):
        if not self.shown:
            self.shown = True
            plt.show()
        
    @staticmethod
    def getMpl():
        if not Mpl.mpl:
            Mpl.mpl = Mpl()
        return Mpl.mpl
    
    @staticmethod
    def cleanup():
        Mpl.mpl = None