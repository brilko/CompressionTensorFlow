import keras
from keras import layers
from numpy import ndarray
import numpy as np
from CodecUtils.CompressedPhoto import CompressedPhoto

from CodecUtils.NeuroNetParameters import NeuroNetParameters

class NeuroNet():
    def __init__(self):
        None

    def encode(self, data: ndarray, neuroNetParameters: NeuroNetParameters, chanels: int):
        p = neuroNetParameters
        encoder = keras.Sequential([layers.Dense(p.tileVolume) 
                                       for _ in range(p.countCompressionLayers)])
        decoder = keras.Sequential([layers.Dense(p.pixelsPerTile * chanels) 
                                         for _ in range(p.countRestoreLayers)])
        model = keras.Sequential([
            encoder,
            decoder
        ])
        model.compile(optimizer='Adam', loss = 'MeanSquaredError')
        model.fit(data, data, epochs = p.countEpochs)
        compressed = encoder.predict(data)
        compressed = np.array(compressed, dtype='float16')
        decoder.compile(optimizer='Adam', loss = 'MeanSquaredError')
        decoder.fit(compressed, data, epochs=p.countEpochs)
        restoreWeights = decoder.weights        
        return (compressed, restoreWeights)
        
    def decode(self, compressedPhoto: CompressedPhoto):
        decoder = keras.Sequential([layers.Dense(compressedPhoto.pixelsPerTile * compressedPhoto.shape[2]) 
                                    for _ in range(compressedPhoto.countDecoderLayers)])
        decoder.build(compressedPhoto.compressedTiles.shape)
        decoder.set_weights(compressedPhoto.weights)
        data = decoder.predict(compressedPhoto.compressedTiles)
        return data
