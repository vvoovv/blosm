

def dumpInput(verts, firstVertIndex, numPolygonVerts, holesInfo, unitVectors):
    with open("D:/tmp/bpypolyskel_test.py", 'w') as file:
        file.write("import mathutils\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("from bpypolyskel import bpypolyskel\n")
        file.write("from collections import Counter\n")
        file.write("\n")
        
        file.write("verts = [\n")
        firstVert = True
        for vert in verts:
            if firstVert:
                firstVert = False
            else:
                file.write(",\n")
            file.write("    mathutils.Vector((%s,%s,%s))" % (vert[0], vert[1], vert[2]))
        file.write("\n]\n")
        
        if unitVectors:
            file.write("unitVectors = [\n")
            firstVert = True
            for unitVector in unitVectors:
                if firstVert:
                    firstVert = False
                else:
                    file.write(",\n")
                file.write("    mathutils.Vector((%s,%s,%s))" % (unitVector[0], unitVector[1], unitVector[2]))
            file.write("\n]\n")
        
        if holesInfo:
            file.write("holesInfo = [\n")
            firstHole = True
            for holeInfo in holesInfo:
                if firstHole:
                    firstHole = False
                else:
                    file.write(",\n")
                file.write("    (%s,%s)" % (holeInfo[0], holeInfo[1]))
            file.write("\n]\n")
        else:
            file.write("holesInfo = None\n")
        
        file.write("firstVertIndex = %s\n" % firstVertIndex)
        file.write("numPolygonVerts = %s\n" % numPolygonVerts)
        if unitVectors:
            file.write("faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)")
        else:
            file.write("faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, None)")
        file.write("\n")
        
        file.write("fig = plt.figure()\n")
        file.write("ax = fig.gca(projection='3d')\n")
        file.write("for face in faces:\n")
        file.write("    for edge in zip(face, face[1:] + face[:1]):\n")
        file.write("        p1 = verts[edge[0]]\n")
        file.write("        p2 = verts[edge[1]]\n")
        file.write("        ax.plot([p1.x,p2.x],[p1.y,p2.y],[p1.z,p2.z])\n")
        file.write("plt.show()\n")