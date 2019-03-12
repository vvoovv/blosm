

class Action:
    
    def __init__(self, app, data, itemStore):
        self.app = app
        self.data = data
        self.itemStore = itemStore
    
    def process(self, building, itemClass):
        pass