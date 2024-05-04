class CompressedPhoto():
    def __init__(self, compressedTiles, weights, shape, pixelsPerTile):
        self.compressedTiles = compressedTiles
        self.weights = weights
        self.shape = shape
        self.pixelsPerTile = pixelsPerTile