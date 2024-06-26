from numpy import ndarray
import numpy as np

class NpImageConverter():
    def __init__(self):
        None
    
    def normalizeNpImage(self, npImage: ndarray, pixelsPerTile: int):
        (height, width, chanels) = npImage.shape
        data = npImage.copy()
        data = data.reshape((round(height*width/pixelsPerTile), pixelsPerTile*chanels))    
        data = data / 255.0
        return data, (height, width, chanels)
    
    def recoverNpImage(self, npImage: ndarray, shape: (int, int, int)):
        recovered = npImage.reshape(shape)
        recovered[recovered<0] = 0
        recovered[recovered>1] = 1
        recovered *= 255
        recovered = recovered.round()
        recovered = np.array(recovered, dtype='uint8')
        return recovered

npImageConverter = NpImageConverter()
