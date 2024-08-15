from .item import Item
from way.item.connectors import IntConnector


class MinorIntersection(Item):
    ID = 0
    def __init__(self, location):
        super().__init__()
        self.id = MinorIntersection.ID
        MinorIntersection.ID += 1

        self.location = location

        self.leftHead = None
        self.leftTail = None
        self.rightHead = None
        self.rightTail = None


    def insertLeftConnector(self, conn):
        # Inserts the instance <connector> of IntConnector at the end of the linear doubly-linked list,
        # attached to self.leftHead. It is inserted "after", which is in counter-clockwise direction.
        connector = conn.copy()
        if self.leftHead is None:
            connector.pred = None
            connector.succ = None
            self.leftHead = connector
            self.leftTail = connector
        else:
            self.leftTail.succ = connector
            connector.succ = None
            connector.pred = self.leftTail
            self.leftTail = connector

    def insertRightConnector(self, conn):
        # Inserts the instance <connector> of IntConnector at the end of the linear doubly-linked list,
        # attached to self.rightHead. It is inserted "after", which is in counter-clockwise direction.
        connector = conn.copy()
        if self.rightHead is None:
            connector.pred = None
            connector.succ = None
            self.rightHead = connector
            self.rightTail = connector
        else:
            self.rightTail.succ = connector
            connector.succ = None
            connector.pred = self.leftTail
            self.rightTail = connector

    @staticmethod
    def iterate_from(conn_item):
        while conn_item is not None:
            yield conn_item
            conn_item = conn_item.succ


