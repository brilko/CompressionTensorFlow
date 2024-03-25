class Configuration():
    def __init__(self):
        self._epochs = 10
        self._pixelsPerTile = 20
        self._midleUnitVolume = 5
        
    @property
    def epochs(self):
        return self._epochs
    
    @property
    def pixelsPerTile(self):
        return self._pixelsPerTile
    
    @property
    def midleUnitVolume(self):
        return self._midleUnitVolume
        
configuration = Configuration()




