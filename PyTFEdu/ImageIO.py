import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy._typing import NDArray

class ImageIO():
    def __init__(self):
        self

    def readImage(self, path: str):
        source = Image.open(path)
        npImage = np.asarray(source)
        print(source.format, source.size, source.mode)
        print("shape is ", npImage.shape)
        print("dtype is ", npImage.dtype)
        print("ndim is ", npImage.ndim)
        print("itemsize is ", npImage.itemsize) 
        print("nbytes is ", npImage.nbytes) 
        return npImage
        
    def saveImage(self, path: str, npImage: NDArray[any]):
        leopardDestiny = Image.fromarray(npImage);
        leopardDestiny.save(path)
        
    def showImage(self, npImage: NDArray[any]):
        plt.imshow(npImage);
        plt.show()

imageIO = ImageIO()

