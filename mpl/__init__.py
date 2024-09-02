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
            # file = 'D:/BLOSM/doc/clustering_02/images/karl_marx_06.png'

            # rotterdam_01
            # plt.xlim([-40,40])
            # plt.ylim([-40,15])
            # plt.xlim([-50,50])
            # plt.ylim([-50,25])
            # file = 'D:/BLOSM/doc/clustering_02/images/rotterdam_06.png'

            # # milano-01
            # plt.xlim([-100,0])
            # plt.ylim([-245,-180])
            # file = 'D:/BLOSM/doc/clustering_02/images/milano_06.png'

            # tokyo_shibuya
            # plt.xlim([-160,-60])
            # plt.ylim([30,110])
            # file = 'D:/BLOSM/doc/clustering_02/images/tokyo_01.png'

            # bratislava
            # plt.xlim([470,570])
            # plt.ylim([150,220])
            # file = 'D:/BLOSM/doc/clustering_02/images/bratislava_06.png'

            # from debug import saveAxisContent
            # saveAxisContent(file)

            plt.gca().axis('equal')
            plt.show()
        
    @staticmethod
    def getMpl():
        if not Mpl.mpl:
            Mpl.mpl = Mpl()
        return Mpl.mpl
    
    @staticmethod
    def cleanup():
        Mpl.mpl = None