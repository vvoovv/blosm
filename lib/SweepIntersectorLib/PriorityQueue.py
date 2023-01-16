# Implementation in pure Python of a sweep line algorithm for line-segment
# intersection, based on the algorithmof described in the paper:
#
# Mehlhorn, K., Näher, S.(1994). Implementation of a sweep line algorithm
# for the Straight Line Segment Intersection Problem (MPI-I-94-160).
# Saarbrücken: Max-Planck-Institut für Informatik.
# https://pure.mpg.de/pubman/faces/ViewItemOverviewPage.jsp?itemId=item_1834220#
# 
# Implementation of the class <PriorityQueue>.
#

import heapq

class QueueNode():
    def __init__(self,key,inf):
        self.key = key
        self.inf = inf
    def __gt__(self,other):
        return self.key > other.key

    def __lt__(self,other):
        return self.key < other.key

    def __ge__(self,other):
        return self.key >= other.key

    def __le__(self,other):
        return self.key <= other.key

    def __eq__(self,other):
        return self.key == other.key

class PriorityQueue():
    def __init__(self):
        self.queue = []

    @staticmethod
    def prio(node):
        """
        Returns the priority of <node>
        """
        return node.key

    @staticmethod
    def inf(node):
        """
        Returns the value of <node>
        """
        return node.inf

    def insert(self, key, value):
        """
        Adds a new node to the structure and returns it.
        """
        node = QueueNode(key,value)
        heapq.heappush(self.queue, node)
        return node

        return self.heapInsert(key, value)

    def min(self):  # corresponds to find_min() in LEDA
        """
        Returns the node with minimal priority 
        None if structure is empty
        """
        if self.queue:
            return self.queue[0]
        else:
            return None

    def delMin(self):
        """
        Removes the node node=self.findMin()
        from structure and return its priority.
        Precondition: the structure is not empty.
        """
        return heapq.heappop(self.queue)

    def size(self):
        """
        Returns the size of the structure
        """
        return len(self.queue)

    # returns True if the structure is empty
    #  and False otherwise.
    def empty(self):
        """
        Returns True if the structure is empty,
        else False.
        """
        return not self.queue
