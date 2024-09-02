minTemplateLength = 2.
minNeighborLength = 2.

transitionLimit = 0.01
transitionSlope = 0.3

dbScanDist = 20.

searchDist = {
        "motorway":      40., "motorway_link":  40., "trunk":      30., "trunk_link":    30., "primary":      30., "primary_link":  30.,
        "secondary":     25., "secondary_link": 25., "tertiary":   20., "tertiary_link": 20., "unclassified": 25., "residential":   15., 
        "living_street": 15., "service":        15., "pedestrian":  5., "track":          5., "escape":       20., "raceway":       20.,        
        "other_roadway": 20., "footway":        10., "path":        5., "cycleway":       5., "bridleway":     5., "rail":          15., 
        "subway":        10., "light_rail":     10., "tram":       20., "funicular":     10., "monorail":     10., "other_railway": 10.,
}

canPairDict = {
    'motorway': {'motorway': True},
    'primary':  {'primary':True, 'secondary':True,'cycleway':True},
    'secondary':  {'secondary':True, 'secondary_link':True, 'primary':True,'cycleway':True},
    'secondary_link': {'secondary_link':True},
    'tertiary': {'tertiary':True,'cycleway':True},
    "unclassified": {'tertiary':True,'cycleway':True,'service':True},
    'residential': {'primary':True, 'secondary':True},
    'cycleway': {'primary':True, 'secondary':True,'tertiary':True},
    # 'footway': {'primary':True, 'secondary':True,'tertiary':True,'cycleway':True},
}

# canPairDict = {
#     'motorway': {'motorway': True},
#     'trunk':    {'trunk': True},
#     'primary':  {'primary':True, 'secondary':True, 'rail':True, 'tram':True},
#     'primary_link': {'primary_link':True},
#     'secondary': {'secondary':True, 'tertiary':True, 'rail':True, 'tram':True},
#     'tertiary': {'tertiary':True, 'tram':True},
#     'tertiary_link': {'tertiary_link':True},
#     'unclassified': {'tram':True},
#     'footway': {'footway':True},
#     'cycleway': {'cycleway':True},
#     'cycleway': {'footway':True},
#     'rail': {'rail':True,'light_rail':True},
#     'tram': {'tram':True},
# }

def canPair(src, dst):
    srcKey = canPairDict.get(src,False)
    if srcKey:
        return srcKey.get(dst,False)
    return False

canPairOld = {
    "motorway": {
                    "motorway":       True, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "motorway_link": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "trunk": {
                    "motorway":      False, "motorway_link":  False, "trunk":       True, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "trunk_link": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "primary": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":       True, "primary_link":  False,
                    "secondary":      True, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":           True, 
                    "subway":        False, "light_rail":     False, "tram":        True, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "primary_link": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":   True,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "secondary": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":      True, "secondary_link": False, "tertiary":    True, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":           True, 
                    "subway":        False, "light_rail":     False, "tram":        True, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "secondary_link": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "tertiary": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":    True, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":        True, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "tertiary_link": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link":  True, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "unclassified": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":        True, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "residential": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "living_street": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "service": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "pedestrian": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "track": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "escape": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "raceway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "other_roadway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "footway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":         True, "path":       False, "cycleway":       True, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "path": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "cycleway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":         True, "path":       False, "cycleway":       True, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "bridleway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "rail": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":           True, 
                    "subway":        False, "light_rail":      True, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "subway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "light_rail": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "tram": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":        True, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "funicular": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "monorail": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
    "other_railway": {
                    "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                    "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                    "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                    "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                    "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                },
}

                #     "motorway":      False, "motorway_link":  False, "trunk":      False, "trunk_link":    False, "primary":      False, "primary_link":  False,
                #     "secondary":     False, "secondary_link": False, "tertiary":   False, "tertiary_link": False, "unclassified": False, "residential":   False, 
                #     "living_street": False, "service":        False, "pedestrian": False, "track":         False, "escape":       False, "raceway":       False,        
                #     "other_roadway": False, "footway":        False, "path":       False, "cycleway":      False, "bridleway":    False, "rail":          False, 
                #     "subway":        False, "light_rail":     False, "tram":       False, "funicular":     False, "monorail":     False, "other_railway": False,
                # },
