from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
from NpImageConverter import npImageConverter as ic
from CompressionModel import compressionModel as model
import pickle
import numpy as np
from CompressedPhoto import CompressedPhoto
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

npImage = iIO.readImage(p.leopard)

data, shape = ic.normalizeNpImage(npImage)

(height, width, chanels) = shape

model.compile(chanels)

model.fit(data)

compressed = model.compressionModel.predict(data)

compressed = np.array(compressed, dtype='float16')

model.restoreModel.compile(optimizer='Adam', loss = 'MeanSquaredError')
model.restoreModel.fit(compressed, data, epochs=c.epochs)

restoreWeights = model.restoreModel.weights

compressedPhoto = CompressedPhoto(compressed, restoreWeights, shape, c.pixelsPerTile)

with open('Encoded/there', 'wb') as f:
    pickle.dump(compressedPhoto, f)
    
with open('Encoded/there', 'rb') as f:
    compressedPhoto: CompressedPhoto = pickle.load(f)

model.compressionModel.weights = compressedPhoto.weights

restored = model.restoreModel.predict(compressed)
recovered = ic.recoverNpImage(restored, shape)

iIO.showImage(recovered)
iIO.saveImage(p.destiny+'best.jpg', recovered)
