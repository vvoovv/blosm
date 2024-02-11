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

        # self.ax.set_axis_off()
        # fig.add_axes(self.ax)

    def show(self):
        if not self.shown:
            self.shown = True
            #Karl-Marx-Allee
            # plt.xlim([400,480])
            # plt.ylim([-150,-80])

            # rotterdam_01
            # plt.xlim([-50,50])
            # plt.ylim([-50,25])

            # corners piestany
            # plt.xlim([-300,-150])
            # plt.ylim([-660,-500])

            # corners milano
            # plt.xlim([-310,-220])
            # plt.ylim([-225,-170])

            # self.ax.set_aspect(1)
            # plt.tight_layout()

            self.ax.axis('equal')
            plt.show()
        
    @staticmethod
    def getMpl():
        if not Mpl.mpl:
            Mpl.mpl = Mpl()
        return Mpl.mpl
    
    @staticmethod
    def cleanup():
        Mpl.mpl = None