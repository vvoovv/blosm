

allWayCategories = set((
    "other",
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
    # "road", # other
    "footway",
    "bridleway",
    "steps",
    "path",
    "cycleway"
))


facadeVisibilityWayCategories = set((
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
))


class Category:
    __slots__ = tuple()
    @staticmethod
    def addCategories():
        for category in allWayCategories:
            setattr(Category, category, category)
Category.addCategories()