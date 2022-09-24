import re

# Primitive scaling to adapt to locales.
# TODO: Get locales later from PML for example.
localScale = 1.

wayCategoryProps = {
    "motorway":         {'default': 12, 'lane': 3.5, 'radius': 10},
    "motorway_link":    {'default': 12, 'lane': 3.5, 'radius': 10},
    "trunk":            {'default': 11, 'lane': 3.5, 'radius': 10},
    "trunk_link":       {'default': 11, 'lane': 3.5, 'radius': 10},
    "primary":          {'default':  8, 'lane': 2.7, 'radius':  4},
    "primary_link":     {'default':  8, 'lane': 2.7, 'radius':  4},
    "secondary":        {'default':  7, 'lane': 2.5, 'radius':  3},
    "secondary_link":   {'default':  7, 'lane': 2.5, 'radius':  3},
    "tertiary":         {'default':  6, 'lane': 2.7, 'radius':  2},
    "tertiary_link":    {'default':  6, 'lane': 2.7, 'radius':  2},
    "residential":      {'default':  7, 'lane': 2.5, 'radius':  2},
    "living_street":    {'default':  5, 'lane': 2.5, 'radius':  2},
    "service":          {'default':  4, 'lane': 2.0, 'radius':  1},
    "pedestrian":       {'default':  2, 'lane': 1.0, 'radius':  1},
    "track":            {'default':  2, 'lane': 1.0, 'radius':0.1},
    "escape":           {'default':  5, 'lane': 3.5, 'radius':  2},
    "raceway":          {'default': 12, 'lane': 3.5, 'radius': 10},
    "footway":          {'default':  2, 'lane': 1.5, 'radius':0.5},
    "path":             {'default':1.5, 'lane': 3.5, 'radius':0.5},
    "cycleway":         {'default':  2, 'lane': 1.5, 'radius':  1},
    "bridleway":        {'default':1.5, 'lane': 1.0, 'radius':  1},
    "unclassified":     {'default':  7, 'lane': 3.5, 'radius':  2},
    "other_roadway":    {'default':  4, 'lane': 2.0, 'radius':  1},
    "rail":             {'default':  3, 'lane': 3.5, 'radius':  1},
    "subway":           {'default':  3, 'lane': 3.5, 'radius':  1},
    "light_rail":       {'default':2.5, 'lane': 3.5, 'radius':  1},
    "tram":             {'default':  3, 'lane': 3.5, 'radius':  1},
    "funicular":        {'default':  3, 'lane': 3.5, 'radius':  1},
    "monorail":         {'default':  3, 'lane': 3.5, 'radius':  1},
    "other_railway":    {'default':  3, 'lane': 3.5, 'radius':  1},
}

intNum   = re.compile('\d*\d+')
floatNum = re.compile('\d*\.?\d+')
alphaNum = re.compile('[A-Za-z]*[A-Za-z]+')

def getWidth(tags):
    attr = tags.get('width',None)
    if not attr: return None
    match = floatNum.search(attr)
    if not match: return None
    width = float(match.group())
    match = alphaNum.search(attr)
    if not match: return width
    if 'cm' in match.group():
        width = width/100
    return width

def getLanes(tags):
    attr = tags.get('lanes',None)
    if not attr: return None
    match = intNum.search(attr)
    if not match: return None
    lanes = int(match.group())
    return lanes

def isOneWay(tags):
    attr = tags.get('oneway',None)
    if not attr: return False
    return attr in ['yes', '-1', 'reversible', 'alternating']

def estimateWayWidth(category,tags):
    width = getWidth(tags)
    if width: return width * localScale

    lanes = getLanes(tags)
    if lanes:
        return lanes * wayCategoryProps[category]['lane'] * localScale

    oneWay = 0.5 if isOneWay else 1.
    return wayCategoryProps[category]['default'] * oneWay * localScale

def estFilletRadius(category,tags):
    return wayCategoryProps[category]['radius']



