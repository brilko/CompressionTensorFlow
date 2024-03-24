from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
from NpImageConverter import npImageConverter as ic
from CompressionModel import compressionModel as model
import pickle
import numpy as np
from CompressedPhoto import CompressedPhoto
from ConfigurationParameters.Configuration import configuration as c

npImage = iIO.readImage(p.source)

data, shape = ic.normalizeNpImage(npImage)

(height, width, chanels) = shape

model.compile(chanels)

model.fit(data)

compressed = model.compressionModel.predict(data)

compressed = np.array(compressed, dtype='float16')

restoreWeights = model.restoreModel.weights

compressedPhoto = CompressedPhoto(compressed, restoreWeights, shape, c.pixelsPerTile)

with open('Encoded/there', 'wb') as f:
    pickle.dump(compressedPhoto, f)
    
with open('Encoded/there', 'rb') as f:
    e = pickle.load(f)

restored = model.restoreModel.predict(compressed)
recovered = ic.recoverNpImage(restored, shape)

iIO.showImage(recovered)

z = 0