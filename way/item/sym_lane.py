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
        self.pred = None
        self.succ = None

        # A reference to the incoming (smaller) Section.
        self.incoming = incoming

        # A reference to the outgoing (wider) Section.
        self.outgoing = outgoing

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
        fwdWidthDiff = self.incoming.forwardWidth - self.outgoing.forwardWidth
        bwdWidthDiff = self.incoming.backwardWidth - self.outgoing.backwardWidth
        transitionLength = max( abs(fwdWidthDiff+bwdWidthDiff)/transitionSlope, 1. ) / 2.

        fwd1, fwd2 = False, False
        if self.incoming.src == self.location:
            fwd1 = True
            tTrans1 = min( self.incoming.polyline.d2t(transitionLength), (len(self.incoming.polyline)-1)/2. )
            self.incoming.trimS = max(self.incoming.trimS,tTrans1)
            p2 = self.incoming.polyline.offsetPointAt(tTrans1,-self.incoming.width/2.)
            p1 = self.incoming.polyline.offsetPointAt(tTrans1,self.incoming.width/2.)
        else:
            tTrans1 = max( self.incoming.polyline.d2t(self.incoming.polyline.length() - transitionLength), (len(self.incoming.polyline)-1)/2. )
            self.incoming.trimT = min(self.incoming.trimT,tTrans1)
            # p2 = way1.polyline.offsetPointAt(tTrans1,-way1.backwardWidth)
            # p1 = way1.polyline.offsetPointAt(tTrans1,way1.forwardWidth)
            p2 = self.incoming.polyline.offsetPointAt(tTrans1,-self.incoming.width/2.)
            p1 = self.incoming.polyline.offsetPointAt(tTrans1,self.incoming.width/2.)

        if self.outgoing.src == self.location:
            fwd2 = True
            tTrans2 = min( self.outgoing.polyline.d2t(transitionLength), (len(self.outgoing.polyline)-1)/2. )
            self.outgoing.trimS = max(self.outgoing.trimS,tTrans2)
            # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.backwardWidth)
            # p4 = way2.polyline.offsetPointAt(tTrans2,way2.forwardWidth)
            p3 = self.outgoing.polyline.offsetPointAt(tTrans2,-self.outgoing.width/2.)
            p4 = self.outgoing.polyline.offsetPointAt(tTrans2,self.outgoing.width/2.)
        else:
            tTrans2 = max( self.outgoing.polyline.d2t(self.outgoing.polyline.length() - transitionLength), (len(self.outgoing.polyline)-1)/2. )
            self.outgoing.trimT = min(self.outgoing.trimT,tTrans2)
            # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.forwardWidth)
            # p4 = way2.polyline.offsetPointAt(tTrans2,way2.backwardWidth)
            p3 = self.outgoing.polyline.offsetPointAt(tTrans2,-self.outgoing.width/2.)
            p4 = self.outgoing.polyline.offsetPointAt(tTrans2,self.outgoing.width/2.)

        self.area = [p1,p2,p3,p4]

        signOfIn = 1 if fwd1 else -1
        signOfOut = 1 if fwd2 else -1
        self.directions = (signOfIn, signOfOut)

