from collections import defaultdict
import queue

class DisjointSets():
    def __init__(self):
        self.G = defaultdict(set)

    def addSegment(self,u,v):
        self.G[u].add(v)
        self.G[v].add(u)

    def bfs(self):
        visited = set()
        for start in self.G:
            if start in visited:
                continue
            connectedNodes = []
            queue = [(start, iter(self.G[start]))]
            while queue:
                parent, children = queue.pop()
                if parent not in visited:
                    connectedNodes.append(parent)
                    visited.add(parent)
                    for child in children:
                        if child not in visited:
                            queue.append((child,self.G[child]))
            yield connectedNodes

    def __iter__(self):
       return self.bfs()



# The code for class ArticulationPoint is adapted from NetworkX (https://networkx.org/).
# Their code is distributed with the 3-clause BSD license (see at end of this file).
class ArticulationPoints():
    def __init__(self):
        self.G = defaultdict(set)
        self.visited = set()
        self.parents = {}
        self.low = {}
        self.res = []

    def addEdge(self, u, v):
        self.G[u].add(v)
        self.G[v].add(u)

    def dfs(self):
        # depth-first search algorithm to generate articulation points
        # and biconnected components
        visited = set()
        for start in self.G:
            if start in visited:
                continue
            discovery = {start: 0}  # time of first discovery of node during search
            low = {start: 0}
            root_children = 0
            visited.add(start)
            edge_stack = []
            stack = [(start, start, iter(self.G[start]))]
            while stack:
                grandparent, parent, children = stack[-1]
                try:
                    child = next(children)
                    if grandparent == child:
                        continue
                    if child in visited:
                        if discovery[child] <= discovery[parent]:  # back edge
                            low[parent] = min(low[parent], discovery[child])
                    else:
                        low[child] = discovery[child] = len(discovery)
                        visited.add(child)
                        stack.append((parent, child, iter(self.G[child])))
                except StopIteration:
                    stack.pop()
                    if len(stack) > 1:
                        if low[parent] >= discovery[grandparent]:
                            yield grandparent
                        low[grandparent] = min(low[parent], low[grandparent])
                    elif stack:  # length 1 so grandparent is root
                        root_children += 1

            # root node is articulation point if it has more than 1 child
            if root_children > 1:
                yield start

    def __iter__(self):
       return self.dfs()

# cc = ConnectedComponents()
# cc.addSegment(1,2)
# cc.addSegment(2,3)
# cc.addSegment(1,4)
# cc.addSegment(5,6)
# cc.addSegment(6,7)
# cc.addSegment(7,8)
# cc.addSegment(9,3)

# for ccc in cc:
#     test=1

# cutVertex = ArticulationPoint()

# cutVertex.addEdge(1,2)
# cutVertex.addEdge(2,3)
# cutVertex.addEdge(3,4)
# # cutVertex.addEdge(4,1)
# cutVertex.addEdge(1,5)
# cutVertex.addEdge(5,6)
# cutVertex.addEdge(6,2)


# for ap in cutVertex.dfs(True):
#     test=1
# # print(ap)
# test=1

# BSD License for NetworkX (https://networkx.org/):
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the NetworkX Developers nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
