from collections import defaultdict
from functools import cmp_to_key

EPSILON = 0.00001

# operator used to compute a counterclockwise or clockwise ordering of the neighbors of a given node.
# v0,v1,v2 are expected to be of type mathutils.Vector with dimension 2
# v0 is the center and v1,v2 are the neighbors.
# it is assumed that the edges v1 - v0 - v2 are not on one line (but this gets not tested!)
def is_ccw(v1, v2, v0):
    d1 = v1.xy - v0.xy
    d2 = v2.xy - v0.xy
    c = d2.cross(d1)
    if c > EPSILON:
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
            ordering = sorted(neighbors, key = cmp_to_key( lambda a,b:is_ccw(vList[a],vList[b],vList[vertex])) )
		
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
                path.append(tup)
                edgeset -= set([tup])
        if (len(path) != 0):
            # convert edge list in vertices list
            vert_list = [e[0] for e in path]
            faces.append(path) 

        final_faces = []
        for face in faces:
            # rotate edge list so that edge of original polygon is first edge
            nextOrigIndex = next(x[0] for x in enumerate(face) if x[1][0]<nrOfPolyVerts and x[1][1]< nrOfPolyVerts)
            face = face[nextOrigIndex:] + face[:nextOrigIndex]
            # convert edge list in vertices list
            vert_list = [e[0] for e in face]
            if any(i >= nrOfPolyVerts for i in vert_list): # exclude polygon and holes
                final_faces.append(vert_list)
        return final_faces

