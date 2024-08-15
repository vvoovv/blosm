class IntConnector():
    ID = 0
    def __init__(self, intersection):
        self.id = IntConnector.ID
        IntConnector.ID += 1

        # the intersection, to which the connector belongs to
        self.intersection = intersection

        # the first index of the connector in the area polygon of Intersection
        self.index = None

        # the item, to which the connector is connected to
        self.item = None

        # the direction of the item, to which the connector is connected to.
        # True, if the item leaves the intersection
        self.leaving = None

        # the preceding connector in the intersection (in clockwise direction)
        self.pred = None

        # the succeeding connector in the intersection (in counter-clockwise direction)
        self.succ = None

    def copy(self):
        conn = IntConnector(self.intersection)
        conn.index = self.index
        conn.item = self.item
        conn.leaving = self.leaving
        conn.pred = self.pred
        conn.succ = self.succ
        return conn

    @staticmethod
    def iterate_from(conn_item):
        start_item = conn_item
        while conn_item is not None:
            yield conn_item
            conn_item = conn_item.succ
            if conn_item == start_item:
                break
