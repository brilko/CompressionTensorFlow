class CompressedPhoto():
    def __init__(self, compressedTiles, weights, countDecoderLayers, shape, pixelsPerTile):
        self.compressedTiles = compressedTiles
        self.weights = weights
        self.countDecoderLayers = countDecoderLayers
        self.shape = shape
        self.pixelsPerTile = pixelsPerTile