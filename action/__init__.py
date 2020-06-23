

class Action:
    
    def __init__(self, app, data, itemStore, itemFactory):
        self.app = app
        self.data = data
        self.itemStore = itemStore
        self.itemFactory = itemFactory
    
    def do(self, building, itemClass, globalRenderer):
        pass