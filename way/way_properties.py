import re
from collections import defaultdict
from difflib import SequenceMatcher

# Primitive scaling to adapt to locales.
# TODO: Get locales later from PML for example.
localScale = 1.

wayCategoryProps = {
    "motorway":         {'default': 12., 'nrLanes': 4, 'lane': 3.5, 'doubleRoadsideWidth': 3.0, 'radius': 10.},
    "motorway_link":    {'default': 12., 'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 10.},
    "trunk":            {'default': 11., 'nrLanes': 4, 'lane': 3.5, 'doubleRoadsideWidth': 3.0, 'radius': 10.},
    "trunk_link":       {'default': 11., 'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 10.},
    "primary":          {'default': 8.,  'nrLanes': 4, 'lane': 2.7, 'doubleRoadsideWidth': 3.0, 'radius': 4.},
    "primary_link":     {'default': 8.,  'nrLanes': 1, 'lane': 2.7, 'doubleRoadsideWidth': 1.0, 'radius': 4.},
    "secondary":        {'default': 7.,  'nrLanes': 2, 'lane': 2.5, 'doubleRoadsideWidth': 1.0, 'radius': 3.},
    "secondary_link":   {'default': 7.,  'nrLanes': 1, 'lane': 2.5, 'doubleRoadsideWidth': 0.5, 'radius': 3.},
    "tertiary":         {'default': 6.,  'nrLanes': 2, 'lane': 2.7, 'doubleRoadsideWidth': 1.0, 'radius': 2.},
    "tertiary_link":    {'default': 6.,  'nrLanes': 1, 'lane': 2.7, 'doubleRoadsideWidth': 0.5, 'radius': 2.},
    "residential":      {'default': 7.,  'nrLanes': 2, 'lane': 2.5, 'doubleRoadsideWidth': 0.5, 'radius': 2.},
    "living_street":    {'default': 5.,  'nrLanes': 2, 'lane': 2.5, 'doubleRoadsideWidth': 0.5, 'radius': 2.},
    "service":          {'default': 4.,  'nrLanes': 1, 'lane': 2.0, 'doubleRoadsideWidth': 0.3, 'radius': 1.},
    "pedestrian":       {'default': 7.,  'nrLanes': 2, 'lane': 2.5, 'doubleRoadsideWidth': 0.5, 'radius': 1.},
    "track":            {'default': 2.,  'nrLanes': 1, 'lane': 1.0, 'doubleRoadsideWidth': 0.5, 'radius': 0.1},
    "escape":           {'default': 5.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 0.5, 'radius': 2.},
    "raceway":          {'default': 12., 'nrLanes': 2, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 10.},
    "footway":          {'default': 2.,  'nrLanes': 1, 'lane': 2.5, 'doubleRoadsideWidth': 0.0, 'radius': 0.5},
    "path":             {'default': 1.5, 'nrLanes': 1, 'lane': 2.5, 'doubleRoadsideWidth': 0.0, 'radius': 0.5},
    "cycleway":         {'default': 2,   'nrLanes': 1, 'lane': 1.5, 'doubleRoadsideWidth': 0.0, 'radius': 1.},
    "bridleway":        {'default': 1.5, 'nrLanes': 1, 'lane': 1.0, 'doubleRoadsideWidth': 0.0, 'radius': 1.},
    "unclassified":     {'default': 7.,  'nrLanes': 2, 'lane': 3.5, 'doubleRoadsideWidth': 0.5, 'radius': 2.},
    "other_roadway":    {'default': 4.,  'nrLanes': 2, 'lane': 2.0, 'doubleRoadsideWidth': 0.5, 'radius': 1.},
    "rail":             {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 1.},
    "subway":           {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 1.},
    "light_rail":       {'default': 2.5, 'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 1.5, 'radius': 1.},
    "tram":             {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 1.5, 'radius': 1.},
    "funicular":        {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 1.},
    "monorail":         {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 1.},
    "other_railway":    {'default': 3.,  'nrLanes': 1, 'lane': 3.5, 'doubleRoadsideWidth': 2.0, 'radius': 1.},
}

# The way width of the following categories is not halved when tagged as one-way.
# See https://github.com/prochitecture/blosm/issues/44#issuecomment-1265216790
halveNrLanesIfOneWay = (
    "motorway", "trunk", "primary"
)

intNum   = re.compile('\d*\d+')
floatNum = re.compile('\d*\.?\d+')
alphaNum = re.compile('[A-Za-z]*[A-Za-z]+')

def getWidth(tags):
    attr = tags.get('width', None)
    if not attr:
        return None
    match = floatNum.search(attr)
    if not match:
        return None
    width = float(match.group())
    match = alphaNum.search(attr)
    if not match:
        return width
    if 'cm' in match.group():
        width = width/100
    return width

def isOneWay(tags):
    attr = tags.get('oneway', None)
    if not attr: return False
    return attr in ['yes', '-1', 'reversible', 'alternating']

# Pattern characters for turn lane tag values.
# N: no move, R: add right, L: add left, r: subtract right, l: subtract left

# Version for right-hand traffic
patternCharRight = {
    # Pattern for lanes of two-way street.
    # Consider left-turn lanes as neutral.
    'two': {
        'none':           'N',
        'through':        'N',
        'reverse':        'N',
        'left':           'N',
        'slight_left':    'N',
        'sharp_left':     'N',
        'right':          'R',
        'slight_right':   'R',
        'sharp_right':    'R',
        'merge_to_left':  'r',
        'merge_to_right': 'l',
    },
    # Pattern for lanes of one-way street.
    # Allow left and right turn lanes.
   'one': {
        'none':           'N',
        'through':        'N',
        'reverse':        'N',
        'left':           'L',
        'slight_left':    'L',
        'sharp_left':     'L',
        'right':          'R',
        'slight_right':   'R',
        'sharp_right':    'R',
        'merge_to_left':  'r',
        'merge_to_right': 'l',
    },
}

# Version for left-hand traffic
patternCharLeft = {
    # Pattern for lanes of two-way street.
    # Consider right-turn lanes as neutral.
    'two': {
        'none':           'N',
        'through':        'N',
        'reverse':        'N',
        'left':           'L',
        'slight_left':    'L',
        'sharp_left':     'L',
        'right':          'N',
        'slight_right':   'N',
        'sharp_right':    'R',
        'merge_to_left':  'r',
        'merge_to_right': 'l',
    },
    # Pattern for lanes of one-way street.
    # Allow left and right turn lanes.
   'one': {
        'none':           'N',
        'through':        'N',
        'reverse':        'N',
        'left':           'L',
        'slight_left':    'L',
        'sharp_left':     'L',
        'right':          'R',
        'slight_right':   'R',
        'sharp_right':    'R',
        'merge_to_left':  'r',
        'merge_to_right': 'l',
    },
}

# Convert turn-lane tag values to lane pattern
# left-hand traffic version
def val2patLeft(tags,fwd):
    pattern = ''
    columns = tags.split('|')
    for col in columns:
        vals = col.split(';')
        if len(vals) > 1:   # for instance 'left;through'
            pattern += 'N'
        elif vals[0]:
            pattern += patternCharRight[fwd][vals[0]]
        else:
            pattern += 'N'
    return pattern
# right-hand traffic version
def val2patRight(tags,fwd):
    pattern = ''
    columns = tags.split('|')
    for col in columns:
        vals = col.split(';')
        if len(vals) > 1:   # for instance 'left;through'
            pattern += 'N'
        elif vals[0]:
            pattern += patternCharLeft[fwd][vals[0]]
        else:
            pattern += 'N'
    return pattern

def lanePattern(category,tags,leftHandTraffic):
    val2pat = val2patLeft if leftHandTraffic else val2patRight
    props = wayCategoryProps[category]
    isOneWay = 'oneway' in tags and tags['oneway'] != 'no'

    fwdPattern = None
    bwdPattern = None
    bothLanes = 0

    if isOneWay:
        if 'lanes' in tags:
            fwdCount = int(tags['lanes'])
        else:
            fwdCount = int(props['nrLanes']/2) if category in halveNrLanesIfOneWay else props['nrLanes']
        fwdTags =  tags['turn:lanes'] if 'turn:lanes' in tags else '|'*(fwdCount-1)
        fwdPattern = val2pat(fwdTags,'one')
        bwdPattern = ''
    else:   # Two-way street.
        # Do we have explicit turn lanes?
        turnTags = {key[5:]:value for key,value in tags.items() if re.search('^turn:', key)}
        if 'lanes:both_ways' in turnTags:
            bothLanes = 1
        if turnTags:
            # Let's see what we can find.
            if 'lanes:forward' in turnTags and 'lanes:backward' in turnTags:
                # Fine, everything is here, we can provide the pattern.
                # The pattern in backward direction has to be reversed.
                fwdPattern = val2pat(turnTags['lanes:forward'],'two')
                bwdPattern = val2pat(turnTags['lanes:backward'],'two')
            elif 'lanes:forward' in turnTags:
                # We need the number of lanes in backward direction
                fwdPattern = val2pat(turnTags['lanes:forward'],'two')
                if 'lanes:backward' in tags:
                    bwdPattern = 'N'*(int(tags['lanes:backward']))
                elif 'lanes' in tags:
                    bwdPattern = 'N'*(int(tags['lanes'])-len(fwdPattern))
                else:
                    # Assume one backward lane
                    bwdPattern = 'N'

            elif 'lanes:backward' in turnTags:
                # We need the number of lanes in forward direction
                bwdPattern = val2pat(turnTags['lanes:backward'],'two')
                if 'lanes:forward' in tags:
                    fwdPattern = 'N'*(int(tags['lanes:forward']))
                elif 'lanes' in tags:
                    fwdPattern = 'N'*(int(tags['lanes'])-len(bwdPattern))
                else:
                    # Assume one forward lane
                    fwdPattern = 'N'

        else: # No turn lanes
            # # Do we have a lane count?
            # if 'lanes' in tags:
            #     laneCount = int(tags['lanes'])
            #     if laneCount > 1:
            #         return isOneWay,'N'*int(laneCount/2),'N'*int(laneCount/2)
            #     else:
            #          return isOneWay,'N', ''
            if 'lanes:forward' in tags and 'lanes:backward' in tags:
                fwdCount = int(tags['lanes:forward'])
                bwdCount = int(tags['lanes:backward'])
                fwdPattern = val2pat('|'*(fwdCount-1),'two')
                bwdPattern = val2pat('|'*(bwdCount-1),'two')
            elif 'lanes:forward' in tags:
                # We need the number of lanes in backward direction
                fwdPattern = 'N'*int(tags['lanes:forward']) 
                if 'lanes' in tags:
                    bwdPattern = 'N'*(int(tags['lanes'])-len(fwdPattern))
                else:
                    # Assume one backward lane
                    bwdPattern = 'N'
            elif 'lanes:backward' in tags:
                # We need the number of lanes in forward direction
                bwdPattern = 'N'*int(tags['lanes:backward'])
                if 'lanes' in tags:
                    fwdPattern = 'N'*(int(tags['lanes'])-len(bwdPattern))
                else:
                    # Assume one forward lane
                    fwdPattern = 'N'
            else:
                if props['nrLanes'] > 1:
                    fwdPattern = 'N'*int(props['nrLanes']/2)
                    bwdPattern = 'N'*int(props['nrLanes']/2)
                else:
                    fwdPattern = 'N'*props['nrLanes']
                    bwdPattern = ''

    return isOneWay,fwdPattern, bwdPattern, bothLanes

def turnsFromPatterns(longer,shorter):
    match = SequenceMatcher(None, longer,shorter).find_longest_match(0, len(longer), 0, len(shorter))
    leftAdded = longer[:match.a]
    rightAdded = longer[(match.a + match.size):]
    # print(shorter, longer, (leftAdded,rightAdded))
    return leftAdded, rightAdded



def estimateWayWidths(section):
    props = wayCategoryProps[section.category]
    width = getWidth(section.tags)

    section.laneWidth = width/section.totalLanes * localScale if width else props['lane'] * localScale

    # roadSideWidth = props['doubleRoadsideWidth']
    # section.width = section.totalLanes * section.laneWidth + roadSideWidth
    # section.forwardWidth =  section.forwardLanes * section.laneWidth + \
    #                         section.bothLanes * section.laneWidth/2. + \
    #                         roadSideWidth/2. if section.forwardLanes else 0.
    # section.backwardWidth = section.backwardLanes * section.laneWidth + \
    #                         section.bothLanes * section.laneWidth/2. + \
    #                         roadSideWidth/2. if section.backwardLanes else 0.

    # doubleRoadsideWidth omitted, see https://github.com/prochitecture/blosm/issues/57#issuecomment-1544025403
    section.width = section.totalLanes * section.laneWidth
    section.forwardWidth =  section.forwardLanes * section.laneWidth + \
                            section.bothLanes * section.laneWidth/2.
    section.backwardWidth = section.backwardLanes * section.laneWidth + \
                            section.bothLanes * section.laneWidth/2.

# def estimateWayWidth(category,tags):
#     props = wayCategoryProps[category]

#     width = getWidth(tags)
#     if width:
#         return width * localScale

#     lanes = getLanes(tags)
#     if lanes:
#         return lanes * props['lane'] * localScale + props['doubleRoadsideWidth']

#     return props['doubleRoadsideWidth'] + props['lane'] * localScale *\
#         (props['nrLanes']/2 if isOneWay(tags) and category in halveNrLanesIfOneWay else props['nrLanes'])

def estFilletRadius(category,tags):
    return wayCategoryProps[category]['radius']



