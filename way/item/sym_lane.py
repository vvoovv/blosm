from .item import Item
from defs.way_cluster_params import transitionSlope

class SymLane(Item):
    # This class describes a transition between two Sections, where one of them has 
    # more lanes additionally to the others. At the transition, a transition area with
    # connectors is constructed. The two Sections are trimmed at this area.
    
    ID = 0
    
    def __init__(self, location, incoming, outgoing):
        super().__init__()
        self.id = SymLane.ID
        SymLane.ID += 1

        self._location = location

        # A reference to the incoming (smaller) Section.
        self.pred = incoming

        # A reference to the outgoing (wider) Section.
        self.succ = outgoing

        # A tuple of direction indicators of the Sections, that connect to this transition.
        # The first indicator refers to the incoming Section and the second to the outgoing
        # Section. The indicator is positive, when the start of the Section connects to the
        # transition in the middle between them, and negative, when the Section ends here.
        self.directions = None

        # The polygon of the transition area
        self.area = []

        self.createSymLane()

    @property
    def location(self):
        return self._location

    def createSymLane(self):
        fwdWidthDiff = self.pred.forwardWidth - self.succ.forwardWidth
        bwdWidthDiff = self.pred.backwardWidth - self.succ.backwardWidth
        transitionLength = max( abs(fwdWidthDiff+bwdWidthDiff)/transitionSlope, 3. ) / 2.

        fwd1, fwd2 = False, False
        if self.pred.src == self.location:
            fwd1 = True
            tTrans1 = min( self.pred.polyline.d2t(transitionLength), (len(self.pred.polyline)-1)/2. )
            self.pred.trimS = max(self.pred.trimS,tTrans1)
            p1 = self.pred.polyline.offsetPointAt(tTrans1,-self.pred.width/2.)
            p2 = self.pred.polyline.offsetPointAt(tTrans1,self.pred.width/2.)
        else:
            tTrans1 = max( self.pred.polyline.d2t(self.pred.polyline.length() - transitionLength), (len(self.pred.polyline)-1)/2. )
            self.pred.trimT = min(self.pred.trimT,tTrans1)
            # p2 = way1.polyline.offsetPointAt(tTrans1,-way1.backwardWidth)
            # p1 = way1.polyline.offsetPointAt(tTrans1,way1.forwardWidth)
            p1 = self.pred.polyline.offsetPointAt(tTrans1,self.pred.width/2.)
            p2 = self.pred.polyline.offsetPointAt(tTrans1,-self.pred.width/2.)

        if self.succ.src == self.location:
            fwd2 = True
            tTrans2 = min( self.succ.polyline.d2t(transitionLength), (len(self.succ.polyline)-1)/2. )
            self.succ.trimS = max(self.succ.trimS,tTrans2)
            # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.backwardWidth)
            # p4 = way2.polyline.offsetPointAt(tTrans2,way2.forwardWidth)
            p3 = self.succ.polyline.offsetPointAt(tTrans2,-self.succ.width/2.)
            p4 = self.succ.polyline.offsetPointAt(tTrans2,self.succ.width/2.)
        else:
            tTrans2 = max( self.succ.polyline.d2t(self.succ.polyline.length() - transitionLength), (len(self.succ.polyline)-1)/2. )
            self.succ.trimT = min(self.succ.trimT,tTrans2)
            # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.forwardWidth)
            # p4 = way2.polyline.offsetPointAt(tTrans2,way2.backwardWidth)
            p3 = self.succ.polyline.offsetPointAt(tTrans2,self.succ.width/2.)
            p4 = self.succ.polyline.offsetPointAt(tTrans2,-self.succ.width/2.)

        self.area = [p1,p2,p3,p4]

        signOfIn = 1 if fwd1 else -1
        signOfOut = 1 if fwd2 else -1
        self.directions = (signOfIn, signOfOut)

