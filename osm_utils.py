def assignTags(obj, tags):
    for key in tags:
        obj[key] = tags[key]


def parse_scalar_and_unit( htag ):
    for i,c in enumerate(htag):
        if not c.isdigit():
            return int(htag[:i]), htag[i:].strip()

    return int(htag), ""
