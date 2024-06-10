from numpy.typing import NDArray
import numpy as np
from CodecUtils.Codec import Codec
from CodecUtils.CompressedPhoto import CompressedPhoto
from CodecUtils.NeuroNetParameters import NeuroNetParameters
from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
import pickle
import time
import os

#	TODO:
#	1) Разные картинки
#	2) Разное количество промежуточных слоёв до сжатия
#	3) Разное количество слоёв при восстановлении
#	4) Разное количество нейронов на выходе модели сжатия

# Нужно записать список картинок из 3 картинок по которым будет происходить процесс сжатия-восстановления
# Для каждой картинки сделать по 3 вида количества промежуточных слоёв
# Для каждой модели модели с нужным количеством слоёв сделать по 3 стоя для восстановления
# Для каждой полученной модели сделать по 3 вида количества нейронов 
# 3 варианта эпох
# Провести расчёты эффективности по потере точности, времени работы, степени сжатия
# Сохранить полученные картинки и метрики 

encodedFile = 'Encoded/there'

class TestTask():
    def __init__(self, imagePath: str, countCompressionLayers: int, 
             countRestoreLayers: int, pixelsPerTile: int, 
             tileVolume: int, epochs: int, pathToImages: str):
        self.imagePath = imagePath
        self.countCompressionLayers = countCompressionLayers
        self.countRestoreLayers = countRestoreLayers
        self.pixelsPerTile = pixelsPerTile 
        self.tileVolume = tileVolume
        self.epochs = epochs
        self.pathToImages = pathToImages

def makeTest(testTask: TestTask):
    t = testTask
    npImage = iIO.readImage(t.imagePath)
    timeStart = time.time()    
    (restored, compressedPhoto) = useCodec(npImage, testTask)
    timeEnd = time.time()
    compression = measureCompression(t.imagePath, compressedPhoto)
    squareResidual = measureSquareResidual(npImage, restored)
    restoredPictureName = f'CompressLayers={t.countCompressionLayers}-RestoreLayers={t.countRestoreLayers}-TilePixels={t.pixelsPerTile}-TileVolume{t.tileVolume}-Epochs={t.epochs}-Time{timeEnd-timeStart}-Compression={compression}-SquareResidual={squareResidual}.bmp'
    iIO.saveImage(t.pathToImages+restoredPictureName, restored)

def useCodec(npImage: NDArray, testTask: TestTask):
    t = testTask
    codec = Codec()
    neuroNetParameters = NeuroNetParameters(t.countCompressionLayers, t.countRestoreLayers,
                                            t.pixelsPerTile, t.tileVolume, t.epochs)
    compressedPhoto = codec.encode(npImage, neuroNetParameters)
    restored = codec.decode(compressedPhoto)
    return (restored, compressedPhoto)
    
def measureCompression(imagePath, compressedPhoto):
    with open(encodedFile, 'wb') as f:
        pickle.dump(compressedPhoto, f)
    originalSize = os.path.getsize(imagePath)
    encodedSize = os.path.getsize(encodedFile)
    return originalSize / encodedSize
    
def measureSquareResidual(original: NDArray, restored: NDArray):
    return (np.square(original - restored)).mean(axis=None)

tileSizes = [10, 20, 40]

def makeTests(pathToOriginal, pathToResults):
    for layers in range(1, 3):
        for tileSize in tileSizes:
            for epochs in range(1, 3):
                testTask = TestTask(pathToOriginal, layers, layers,
                                    tileSize, tileSize/2, epochs, pathToResults)
                makeTest(testTask)

makeTests(p.leopardSource, p.leopardResultDirectory)
makeTests(p.railwaySource, p.railwayResultDirectory)
makeTests(p.skyskebSource, p.skyskebResultDirectory)