from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
from NpImageConverter import npImageConverter as ic
from CompressionModel import codec
import pickle
from CompressedPhoto import CompressedPhoto

with open('Encoded/there', 'rb') as f:
    compressedPhoto: CompressedPhoto = pickle.load(f)

codec.compile(compressedPhoto.shape[2])
codec.encoder.compile(optimizer='Adam', loss = 'MeanSquaredError')

a = compressedPhoto.compressedTiles.shape[1]

codec.encoder.build((1, a))

codec.encoder.set_weights(compressedPhoto.weights)

restored = codec.encoder.predict(compressedPhoto.compressedTiles)
recovered = ic.recoverNpImage(restored, compressedPhoto.shape)

iIO.showImage(recovered)
iIO.saveImage(p.pathToImages+'best.bmp', recovered)
