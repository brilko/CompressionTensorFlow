import keras
from keras import layers
from numpy import ndarray

class NeuroNet():
    def __init__(self):
        None
    
    def compile(self, tileVolume: int, midleUnitVolume: int):
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

    def fit(self, data: ndarray, epochs: int = 2):
        self.model.fit(data, data, epochs=epochs)

    def predict(self, data: ndarray):
        return self.model.predict(data)

neuroNet = NeuroNet()