from ConfigurationParameters.Pathes import path as p
from ImageIO import imageIO as iIO
from NpImageConverter import npImageConverter as ic
from CompressionModel import compressionModel as model

npImage = iIO.readImage(p.source)

data, shape = ic.normalizeNpImage(npImage)

(height, width, chanels) = shape

model.compile(chanels)

model.fit(data)

q = model.compressionModel.predict(data)
w = model.restoreModel.predict(q)
recovered = ic.recoverNpImage(w, shape)

iIO.showImage(recovered)

z = 0