from collections import defaultdict
from functools import cmp_to_key

EPSILON = 0.00001

# Adapted from https://stackoverflow.com/questions/16542042/fastest-way-to-sort-vectors-by-angle-without-actually-computing-that-angle
# Input:  d: difference vector.
# Output: a number from the range [0 .. 4] which is monotonic
#         in the angle this vector makes against the x axis.
def pseudoangle(d):
    p = d[0]/(abs(d[0])+abs(d[1])) # -1 .. 1 increasing with x
    if d[1] < 0: 
        return 3 + p  #  2 .. 4 increasing with x
    else:
        return 1 - p  #  0 .. 2 decreasing with x

def compare_angles(vList,p1,p2,center):
    a1 = pseudoangle(vList[p1] - vList[center])
    a2 = pseudoangle(vList[p2] - vList[center])
    if a1<a2:
        return 1
    else:
        return -1

class poly2FacesGraph:
    def __init__(self):
        self.g_dict = {}

    def add_vertex(self, vertex):
    # if vertex not yet known, add empty list
        if vertex not in self.g_dict:
            self.g_dict[vertex] = []

    def add_edge(self, edge):
    # edge of type set, tuple or list,
    # loops are not allowed, but not tested. 
        edge = set(edge)
        if len(edge) == 2:  # exclude loops
            vertex1 = edge.pop()
            vertex2 = edge.pop()
            self.add_vertex(vertex1)
            self.add_vertex(vertex2)
            self.g_dict[vertex1].append(vertex2)
            self.g_dict[vertex2].append(vertex1)

    def edges(self):
    # returns the edges of the graph
        edges = []
        for vertex in self.g_dict:
            for neighbour in self.g_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append((vertex, neighbour))
        return edges

    def circular_embedding(self, vList, direction = 'CCW'):
        embedding = defaultdict(list)

        for vertex in self.g_dict:
            neighbors = (self.g_dict[vertex])
            ordering = sorted(neighbors, key = cmp_to_key( lambda a,b: compare_angles(vList,a,b,vertex)) )
		
            if direction == 'CCW':  # counter-clockwise
                embedding[vertex] = ordering	
            elif direction == 'CW': # clockwise
                embedding[vertex] = ordering[::-1]

        return embedding

    def faces(self, embedding, nrOfPolyVerts):
        # adapted from SAGE's trace_faces

        # establish set of possible edges
        edgeset = set([])
        for edge in self.edges():
            edgeset = edgeset.union( set([(edge[0],edge[1]),(edge[1],edge[0])]))

        # storage for face paths
        faces = []
        path = []
        face_id = 0
        for edge in edgeset:
            path.append(edge)
            edgeset -= set([edge])
            break  # (Only one iteration)

        # Trace faces
        while (len(edgeset) > 0):
            neighbors = embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2])+1)%(len(neighbors))]
            tup = (path[-1][-1],next_node)
            if tup == path[0]:
                # convert edge list in vertices list
                vert_list = [e[0] for e in path]
                faces.append(path)
                face_id += 1
                path = []
                for edge in edgeset:
                    path.append(edge)
                    edgeset -= set([edge])
                    break  # (Only one iteration)
            else:
                if tup in path:
                    raise Exception('Endless loop catched in poly2FacesGraph faces()') 
                path.append(tup)
                edgeset -= set([tup])
        if (len(path) != 0):
            # convert edge list in vertices list
            vert_list = [e[0] for e in path]
            faces.append(path) 

        final_faces = []
        for face in faces:
            # rotate edge list so that edge of original polygon is first edge
            origEdges = [x[0] for x in enumerate(face) if x[1][0]<nrOfPolyVerts and x[1][1]< nrOfPolyVerts]
            # if no result: face is floating without polygon contour edges
            if origEdges:
                nextOrigIndex = next(x[0] for x in enumerate(face) if x[1][0]<nrOfPolyVerts and x[1][1]< nrOfPolyVerts)
                face = face[nextOrigIndex:] + face[:nextOrigIndex]
            # convert edge list in vertices list
            vert_list = [e[0] for e in face]
            if any(i >= nrOfPolyVerts for i in vert_list): # exclude polygon and holes
                final_faces.append(vert_list)
        return final_faces

