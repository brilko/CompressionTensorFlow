from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
from CodecUtils.NpImageConverter import npImageConverter as ic
from CodecUtils.NeuroNet import neuroNet
import pickle
import numpy as np
from CodecUtils.CompressedPhoto import CompressedPhoto
from ConfigurationParameters.Configuration import configuration as c

#	TODO:
#	1) Разные картинки
#	2) Разное количество промежуточных слоёв до сжатия
#	3) Разное количество слоёв при восстановлении
#	4) Разное количество нейронов на выходе модели сжатия

# Нужно записать список картинок из 3 картинок по которым будет происходить процесс сжатия-восстановления
# Для каждой картинки сделать по 3 вида количества промежуточных слоёв
# Для каждой модели модели с нужным количеством слоёв сделать по 3 стоя для восстановления
# Для каждой полученной модели сделать по 3 вида количества нейронов на выходе
# Провести расчёты эффективности по потере точности, времени работы, степени сжатия
# Сохранить полученные картинки и метрики 

npImage = iIO.readImage(p.leopardSource)

data, shape = ic.normalizeNpImage(npImage)

(height, width, chanels) = shape

neuroNet.compile(chanels)

neuroNet.fit(data)

compressed = neuroNet.coder.predict(data)

compressed = np.array(compressed, dtype='float16')

neuroNet.encoder.compile(optimizer='Adam', loss = 'MeanSquaredError')
neuroNet.encoder.fit(compressed, data, epochs=c.epochs)

restoreWeights = neuroNet.encoder.weights

compressedPhoto = CompressedPhoto(compressed, restoreWeights, shape, c.pixelsPerTile)

with open('Encoded/there', 'wb') as f:
    pickle.dump(compressedPhoto, f)
    
with open('Encoded/there', 'rb') as f:
    compressedPhoto: CompressedPhoto = pickle.load(f)

neuroNet.encoder.set_weights(compressedPhoto.weights)

restored = neuroNet.encoder.predict(compressedPhoto.compressedTiles)
recovered = ic.recoverNpImage(restored, compressedPhoto.shape)

iIO.showImage(recovered)
iIO.saveImage(p.pathToImages+'best.bmp', recovered)
