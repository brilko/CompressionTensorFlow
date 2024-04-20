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

data = data * 255
data = data.round()
data = np.array(data, dtype='uint8')
class treeNode():
    def __init__(self, chanel: int):
        self.chanel = chanel
        self.count = 1
        self.children: list[treeNode] = []
        
    def toChild(self, childChanel: int):
        self.count += 1
        for child in self.children:
            if child.chanel == childChanel:
                return child
        newChild = treeNode(childChanel)
        self.children.append(newChild)
        return newChild
        
tree = treeNode(-1)
branch = tree
for tile in data:
    branch = tree
    for chanel in tile:
        branch = branch.toChild(chanel)

    

recovered = ic.recoverNpImage(restored, shape)

iIO.showImage(recovered)
# iIO.saveImage(p.destiny+'best.jpg', recovered)

z = 0



# model.compile(chanels)

# model.fit(data)

# compressed = model.compressionModel.predict(data)

# compressed = np.array(compressed, dtype='float16')

# model.restoreModel.compile(optimizer='Adam',
#                   loss = 'MeanSquaredError')
# model.restoreModel.fit(compressed, data, epochs=c.epochs)

# restoreWeights = model.restoreModel.weights

# compressedPhoto = CompressedPhoto(compressed, restoreWeights, shape, c.pixelsPerTile)

# with open('Encoded/there', 'wb') as f:
#     pickle.dump(compressedPhoto, f)
    
# with open('Encoded/there', 'rb') as f:
#     e = pickle.load(f)

# restored = model.restoreModel.predict(compressed)