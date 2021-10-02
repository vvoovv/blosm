

allRoadwayCategories = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian",
    "track",
    "escape",
    "raceway",
    "other",
    # "road", # other
    "steps",
    "footway",
    "path",
    "cycleway",
    "bridleway"
)
allRoadwayCategoriesSet = set(allRoadwayCategories)


allRailwayCategories = (
    "rail",
    "subway",
    "light_rail",
    "tram",
    "funicular",
    "monorail"
    
)
allRailwayCategoriesSet = set(allRailwayCategories)


allWayCategories = allRoadwayCategories + allRailwayCategories


facadeVisibilityWayCategories = (
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian"
    #"track"
    #"footway",
    #"steps",
    #"cycleway"
)
facadeVisibilityWayCategoriesSet = set(facadeVisibilityWayCategories)


mainRoads =   (  
    "primary",
    # "primary_link",
    "secondary",
    # "secondary_link",
    "tertiary",
    "residential"
)

smallRoads = (
    #"residential",
    "service",
    # "pedestrian",
    # "track",
    # "escape",
    "footway",
    # "bridleway",
    # "steps",
    # "path",
    "cycleway"
)


class Category:
    __slots__ = tuple()
    @staticmethod
    def addCategories():
        for category in allWayCategories:
            setattr(Category, category, category)
Category.addCategories()