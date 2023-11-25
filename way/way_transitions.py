from itertools import tee,cycle,islice
from defs.way_cluster_params import transitionSlope
from way.way_properties import turnsFromPatterns

# from osmPlot import *

# helper functions -----------------------------------------------
def cycleTriples(iterable):
    # iterable -> (pn-1,pn,p0), (pn,p0,p1), (p0,p1,p2), (p1,p2,p3), (p2,p3,p4), ... 
    p1, p2, p3 = tee(iterable,3)
    p1 = islice(cycle(p1), len(iterable) - 2, None)
    p2 = islice(cycle(p2), len(iterable) - 1, None)
    return zip(p1,p2,p3)
# ----------------------------------------------------------------
# def printTags(way,typ=''):
#     lanes = way.tags['lanes'] if 'lanes' in way.tags else 0
#     forward = way.tags['lanes:forward'] if 'lanes:forward' in way.tags else 0
#     backward = way.tags['lanes:backward'] if 'lanes:backward' in way.tags else 0
#     turns = way.tags['turn:lanes'] if 'turn:lanes' in way.tags else ''
#     turns_fwd = way.tags['turn:lanes:forward'] if 'turn:lanes:forward' in way.tags else ''
#     turns_bwd = way.tags['turn:lanes:backward'] if 'turn:lanes:backward' in way.tags else ''
#     patterns = way.lanePatterns
#     print( '%s: L:%d F:%d B%d T:%s TF:%s TB:%s  P: %s\n' %(typ, int(lanes), int(forward), int(backward), (turns), (turns_fwd), (turns_bwd), patterns) )


def createSideLaneData(node,way1,way2):
    preTurnWay, turnWay = (way1, way2) if way1.totalLanes < way2.totalLanes else (way2, way1)

    # Way width and offset correction for seamless connections according to
    # https://github.com/prochitecture/blosm/issues/57#issuecomment-1544025403
    widePixels = turnWay.totalLanes*220. + 8*(turnWay.totalLanes-1)
    smallPixels = preTurnWay.totalLanes*220. + 8*(turnWay.totalLanes-2)
    factor = widePixels / smallPixels
    # Fix ratio by changing wider way
    turnWay.width = factor * preTurnWay.width
    turnWay.laneWidth = turnWay.width / turnWay.totalLanes

    # Create IDs
    signOfPre = 1 if preTurnWay.originalSection.s == node else -1
    signOfTurn = 1 if turnWay.originalSection.s == node else -1
    wayIDs = (signOfPre*preTurnWay.id,signOfTurn*turnWay.id)
    # printTags(preTurnWay,'pre')
    # printTags(turnWay,'turn')

    # Determine turn types and offset
    if preTurnWay.isOneWay:
        fwdL, fwdR = turnsFromPatterns(turnWay.lanePatterns[0],preTurnWay.lanePatterns[0])
        laneL = bool(fwdL)
        laneR = bool(fwdR)
        turnWay.fwdLaneR = fwdR
        turnWay.fwdLaneL = fwdL
        if laneL or laneR:
            turnWay.offset = turnWay.laneWidth/2. if laneR else -turnWay.laneWidth/2.
    else: # two-ways
        fwdL, fwdR = turnsFromPatterns(turnWay.lanePatterns[0],preTurnWay.lanePatterns[0])
        bwdL, bwdR = turnsFromPatterns(turnWay.lanePatterns[1],preTurnWay.lanePatterns[1])
        laneL = bool(fwdL) or bool(bwdL)
        laneR = bool(fwdR) or bool(bwdR)
        turnWay.fwdLaneR = fwdR
        turnWay.fwdLaneL = fwdL
        turnWay.bwdLaneR = bwdR
        turnWay.bwdLaneL = bwdL
        if laneL or laneR:
            turnWay.offset = turnWay.laneWidth/2. if laneR else -turnWay.laneWidth/2.
    return wayIDs, laneL, laneR

def createSymLaneData(node,way1,way2):
    fwdWidthDiff = way1.forwardWidth - way2.forwardWidth
    bwdWidthDiff = way1.backwardWidth - way2.backwardWidth
    transitionLength = max( abs(fwdWidthDiff+bwdWidthDiff)/transitionSlope, 1. ) / 2.
 
    area = []
    fwd1, fwd2 = False, False
    if way1.originalSection.s == node:
        fwd1 = True
        tTrans1 = min( way1.polyline.d2t(transitionLength), (len(way1.polyline)-1)/2. )
        way1.trimS = max(way1.trimS,tTrans1)
        # p2 = way1.polyline.offsetPointAt(tTrans1,-way1.forwardWidth)
        # p1 = way1.polyline.offsetPointAt(tTrans1,way1.backwardWidth)
        p2 = way1.polyline.offsetPointAt(tTrans1,-way1.width/2.)
        p1 = way1.polyline.offsetPointAt(tTrans1,way1.width/2.)
    else:
        tTrans1 = max( way1.polyline.d2t(way1.polyline.length() - transitionLength), (len(way1.polyline)-1)/2. )
        way1.trimT = min(way1.trimT,tTrans1)
        # p2 = way1.polyline.offsetPointAt(tTrans1,-way1.backwardWidth)
        # p1 = way1.polyline.offsetPointAt(tTrans1,way1.forwardWidth)
        p2 = way1.polyline.offsetPointAt(tTrans1,-way1.width/2.)
        p1 = way1.polyline.offsetPointAt(tTrans1,way1.width/2.)

    if way2.originalSection.s == node:
        fwd2 = True
        tTrans2 = min( way2.polyline.d2t(transitionLength), (len(way2.polyline)-1)/2. )
        way2.trimS = max(way2.trimS,tTrans2)
        # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.backwardWidth)
        # p4 = way2.polyline.offsetPointAt(tTrans2,way2.forwardWidth)
        p3 = way2.polyline.offsetPointAt(tTrans2,-way2.width/2.)
        p4 = way2.polyline.offsetPointAt(tTrans2,way2.width/2.)
    else:
        tTrans2 = max( way2.polyline.d2t(way2.polyline.length() - transitionLength), (len(way2.polyline)-1)/2. )
        way2.trimT = min(way2.trimT,tTrans2)
        # p3 = way2.polyline.offsetPointAt(tTrans2,-way2.forwardWidth)
        # p4 = way2.polyline.offsetPointAt(tTrans2,way2.backwardWidth)
        p3 = way2.polyline.offsetPointAt(tTrans2,-way2.width/2.)
        p4 = way2.polyline.offsetPointAt(tTrans2,way2.width/2.)

    area = [p1,p2,p3,p4]
    # plotWay(way1.polyline,way1.forwardWidth,way1.width/2.,'k')
    # plotWay(way2.polyline,way2.forwardWidth,way2.width/2.,'k')
    # plotPolygon(area,True,'r','r',1,True)
    # plotEnd()

    clustConnectors = dict()
    clustConnectors[way1.id if fwd1 else -way1.id] = 0
    clustConnectors[way2.id if fwd2 else -way2.id] = 2
    return area, clustConnectors
 