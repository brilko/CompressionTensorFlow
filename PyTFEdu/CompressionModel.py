import keras
from keras import layers
from numpy import ndarray
from ConfigurationParameters.Configuration import configuration as c

class Codec():
    def __init__(self):
        None
    
    def compile(self, chanels: int, midleUnitVolume: int = c.midleUnitVolume):
        tileVolume = c.pixelsPerTile * chanels
        self.coder = keras.Sequential([
            layers.Dense(midleUnitVolume),
        ])
        self.encoder = keras.Sequential([    
            layers.Dense(tileVolume),
            layers.Dense(tileVolume),
        ])
        self.model = keras.Sequential([
            self.coder,
            self.encoder
        ])
        self.model.compile(optimizer='Adam',
                  loss = 'MeanSquaredError')

    def fit(self, data: ndarray):
        self.model.fit(data, data, epochs=c.epochs)

    def predict(self, data: ndarray):
        return self.model.predict(data)

codec = Codec()