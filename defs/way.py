

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
    "other_roadway",
    # "road", # other
    "steps",
    "footway",
    "path",
    "cycleway",
    "bridleway"
)
allRoadwayCategoriesSet = set(allRoadwayCategories)

# hierarchy of way categories
allRoadwayCategoriesRank = dict(zip(allRoadwayCategories, range(len(allRoadwayCategories))))

allRailwayCategories = (
    "rail",
    "subway",
    "light_rail",
    "tram",
    "funicular",
    "monorail",
    "other_railway"
)
allRailwayCategoriesSet = set(allRailwayCategories)

allWayCategories = allRoadwayCategories + allRailwayCategories

roadwayIntersectionCategories  = (
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
    "other_roadway",
    "footway",
    "path",
    "cycleway",
    "bridleway"
)
roadwayIntersectionCategoriesSet = set(roadwayIntersectionCategories)

railwayIntersectionCategories = (
    "rail",
    "subway",
    "light_rail",
    "tram",
    "funicular",
    "monorail",
    "other_railway"
)
railwayIntersectionCategoriesSet = set(railwayIntersectionCategories)

wayIntersectionCategories = roadwayIntersectionCategories + railwayIntersectionCategories

allWayIntersectionCategoriesRank = dict(zip(wayIntersectionCategories, range(len(wayIntersectionCategories))))

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
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
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