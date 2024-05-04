class CompressedPhoto():
    def __init__(self, compressedTiles, weights, countEncoderLayers, shape, pixelsPerTile):
        self.compressedTiles = compressedTiles
        self.weights = weights
        self.countEncoderLayers = countEncoderLayers
        self.shape = shape
        self.pixelsPerTile = pixelsPerTile