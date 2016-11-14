

def assignTags(obj, tags):
    for key in tags:
        obj[key] = tags[key]


def parseNumber(s, defaultValue=None):
    try:
        n = float(s)
    except ValueError:
        n = defaultValue
    return n
