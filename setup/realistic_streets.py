_categoryToStreetStyle = {
    "motorway": "motorway",
    "motorway_link": "motorway_link",
    "trunk": "motorway",
    "trunk_link": "motorway_link",
    "primary": "primary",
    "primary_link": "primary_link",
    "secondary": "secondary",
    "secondary_link": "secondary_link",
    "tertiary": "secondary",
    "tertiary_link": "secondary_link",
    "residential": "residential",
    "living_street": "residential",
    "service": "residential",
    "pedestrian": "residential"
}

def getStyleStreet(street):
    return _categoryToStreetStyle.get(street.getMainCategory(), "residential")