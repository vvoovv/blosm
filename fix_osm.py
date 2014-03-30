import xml.etree.cElementTree as etree

doc = etree.parse("tula1891.osm")
osm = doc.getroot()

elementsToRemove = []

for e in osm:
	if "action" in e.attrib:
		if e.attrib["action"] == "delete":
			elementsToRemove.append(e)
		del e.attrib["action"]

for e in elementsToRemove:
	osm.remove(e)

doc.write("test2.osm", encoding="utf8")
