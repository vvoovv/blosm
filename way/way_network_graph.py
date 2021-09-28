# Heavily adapted from 'OsmToRoadGraph' by Andreas AndGem (https://github.com/AndGem/OsmToRoadGraph)
#
# His License: MIT License
# Copyright (c) 2017 AndGem
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import deque, defaultdict
import numpy as np

class OSMWay():
    def __init__(self, id, name, category, nodes, length, forward=True, backward=True):
        self.id = id
        self.name = name
        self.category = category
        self.nodes = nodes
        self.length = length
        self.forward = forward
        self.backward = backward

class Vertex():
    ID = 0
    def __init__(self, id, osm_id, data):
        self.id = id
        self.osm_id = osm_id
        self.data = data
        self.iID = Vertex.ID
        Vertex.ID += 1

class Edge:
    ID = 0
    def __init__(self, s, t, length, geom_dist, category, name, forward, backward):
        self.s = s
        self.t = t
        self.length = length
        self.geom_dist = geom_dist
        self.category = category
        self.name = name
        self.forward = forward
        self.backward = backward
        self.iID = Edge.ID
        Edge.ID += 1

class WayNetworkGraph():
    def __init__(self, nodes, ways):
        self.edges = []
        self.vertices = []
        self.out_ways = []
        self.in_ways = []
        self.way_categories = []

        node_ids = nodes.keys()
        id_mapper = dict(zip(node_ids, range(len(node_ids))))
        if type(ways[0]) is OSMWay:
            # input comes from OSM ways
            self.add_nodes(id_mapper, nodes)
            self.add_osm_edges(id_mapper, ways)
        else:
            # input comes from SectionGraphCreator
            for ID, vertex in nodes.items():
                self.vertices.append(vertex)
                self.out_ways.append([])
                self.in_ways.append([])
                self.way_categories.append([])
            for e in ways:
                e.s = id_mapper[e.s]
                e.t = id_mapper[e.t]
                self.add_edge(e)


    def add_nodes(self, id_mapper, nodes):
        for osm_id, n in nodes.items():
            self.add_node(Vertex(id_mapper[osm_id],osm_id, n))
        test = 1

    def add_node(self, vertex):
        self.vertices.append(vertex)
        self.out_ways.append([])
        self.in_ways.append([])
        self.way_categories.append([])

    def add_osm_edges(self, id_mapper, ways):
        bidirectional_ways = {}
        for w in ways:
            for i in range(len(w.nodes) - 1):
                s_id, t_id = id_mapper[w.nodes[i]], id_mapper[w.nodes[i + 1]]
                segment_length = np.linalg.norm(self.vertices[s_id].data - self.vertices[t_id].data)
                edge = Edge(s_id, t_id, segment_length, segment_length, w.category, w.name, w.forward, w.backward)
                if w.forward and w.backward:
                    smaller, bigger = min(s_id, t_id), max(s_id, t_id)
                    if (smaller, bigger) in bidirectional_ways:
                        # skip duplicated bidirectional edge
                        continue
                    bidirectional_ways[(smaller, bigger)] = w.id

                self.add_edge(edge)

    def add_edge(self,edge):
        self.edges.append(edge)        
        if edge.forward:
            self.out_ways[edge.s].append(edge.t)
            self.way_categories[edge.s].append(edge.category)
            self.in_ways[edge.t].append(edge.s)
        if edge.backward:
            self.out_ways[edge.t].append(edge.s)
            self.way_categories[edge.t].append(edge.category)
            self.in_ways[edge.s].append(edge.t)

    def all_neighbors(self, node_id):
        return list(
            set(self.out_ways[node_id]).union(set(self.in_ways[node_id]))
        )

    def get_out_edges(self):
        result = defaultdict(list)
        for e in self.edges:
            if e.forward:
                result[e.s].append(e)
            if e.backward:
                result[e.t].append(e)
        return result

    def get_node(self, node_id):
        return self.vertices[node_id]

    def get_crossings_that_contain(self, categories):
        found = []
        categories_set = set(categories)
        for indx, v in enumerate(self.vertices):
            v_cats = self.way_categories[indx]
            cats_set = set(v_cats)
            degree = len(v_cats)
            if degree != 2 or (degree==2 and len(cats_set) > 1):
                if categories_set & cats_set:
                    found.append(indx)
        return found



class SectionGraphCreator:
    def __init__(self, graph):
        self.graph = graph

    def createSectionNetwork(self):
        self.out_ways_per_node = self.get_out_ways()
        way_sections = self.find_way_sections()
        node_ids = self.gather_node_ids(way_sections)
        nodes = self.get_nodes(node_ids)
        return WayNetworkGraph(nodes, way_sections)
 
    def find_way_sections(self):
        # Intialize a container (deque) with all nodes that are intersections or ends (degree != 2)
        # Call it <start_nodes>
        # For each node in this <start_nodes>, given by its index <node_id>:
        #     Find all outgoing edges as <out_edges>
        #     For each edge <out_edge> of these outgoing edges, starting by <start_node>:
        #         Follow the neighbors of the way started by <out_edge>, until an <end_node> is found:
        #             1) another intersection node is found
        #             2) an edge is encountered on the way that is different (different category, name, direction)
        #             3) the start_node is found => loop (remove completely)
        #             The edges from <start_node> to <end_node> get merged to a <way_section>
        self.start_nodes = self.find_all_intersections()
        self.seen_start_nodes = set(self.start_nodes)
        self.out_edges_per_node = self.get_out_ways()

        bidirectional_ways = set()
        way_sections = []
        while len(self.start_nodes) > 0:
            node_id = self.start_nodes.popleft()
            out_edges = self.out_edges_per_node[node_id]

            for out_edge in out_edges:
                start_node_id = node_id
                e_merge, end_node_id = self.find_edges_to_merge( start_node_id, out_edge)
                if len(e_merge) == 0:
                    continue

                e_merge[0].length = sum([e.length for e in e_merge])
                e_merge[0].geom_dist = np.linalg.norm(self.graph.vertices[end_node_id].data-self.graph.vertices[start_node_id].data)
                if e_merge[0].backward:
                    #  deduplication measure; if not for this for bidirectional edges, that are
                    #  removed between intersections, 2 new edges would be created
                    lo_node_id, hi_node_id = min(start_node_id, end_node_id), max(start_node_id, end_node_id)
                    if (lo_node_id, hi_node_id) in bidirectional_ways:
                        # already added this edge skip it
                        continue
                    bidirectional_ways.add((lo_node_id, hi_node_id))
                    merged_edge = Edge(lo_node_id, hi_node_id, e_merge[0].length, e_merge[0].geom_dist,
                                        e_merge[0].category, e_merge[0].name, True, e_merge[0].backward)
                else:
                    merged_edge = Edge(start_node_id, end_node_id, e_merge[0].length, e_merge[0].geom_dist,
                                       e_merge[0].category, e_merge[0].name, True, e_merge[0].backward)
                way_sections.append(merged_edge)
        return way_sections

    def get_out_ways(self):
        return self.graph.get_out_edges()

    def find_all_intersections(self):
        node_ids = range(0, len(self.graph.vertices))
        return deque(filter(self.is_intersection, node_ids))

    def get_nodes(self, node_ids):
        return {id : self.graph.vertices[id] for id in node_ids}

    def gather_node_ids(self, edges):
        node_ids = set()
        for e in edges:
            node_ids.add(e.s)
            node_ids.add(e.t)
        return node_ids

    def is_intersection(self, node_id: int):
        return len(self.graph.all_neighbors(node_id)) != 2

    def find_edges_to_merge(self, start_node_id, first_out_edge):
        # Follow the neighbors of the way started by <out_edge>, until an <end_node> is found:
        #     1) another intersection node is found
        #     2) an edge is encountered on the way that is different (different category, name, direction)
        #     3) the start_node is found => loop (remove completely)
        #     4) the end of a way is found
        used_edges = []
        out_edge = first_out_edge
        current_node_id = start_node_id
        while True:
            used_edges.append(out_edge)
            next_node_id = out_edge.t if out_edge.s == current_node_id else out_edge.s

            # 3) the start_node is found => loop (remove completely)
            if next_node_id == start_node_id:
                # detected a loop => remove it
                used_edges = []
                break

            # 1) another intersection node is found
            if self.is_intersection(next_node_id):
                break

            # 2) an edge is encountered on the way that is different (different category, name, direction)
            next_out_edges = list(
                filter(
                    lambda e: current_node_id not in (e.s, e.t),
                    self.out_edges_per_node[next_node_id],
                )
            )

            # 4) the end of a way is found
            if len(next_out_edges) == 0:
                # detected a dead end => stop
                break

            if len(next_out_edges) > 1:
                # something is wrong.. this should have been filtered out by the intersection check
                assert False

            next_out_edge = next_out_edges[0]
            if self.is_not_same_edge(out_edge, next_out_edge):
                if next_node_id not in self.seen_start_nodes:
                    # found a new possible start node
                    self.seen_start_nodes.add(next_node_id)
                    self.start_nodes.append(next_node_id)
                # break since we need to stop here
                break

            out_edge = next_out_edge
            current_node_id = next_node_id

        final_node_id = next_node_id
        return used_edges, final_node_id

    def is_not_same_edge(self, e1: Edge, e2: Edge):
        return (
            e1.category != e2.category
            # or e1.name != e2.name
            or e1.backward != e2.backward
        )


