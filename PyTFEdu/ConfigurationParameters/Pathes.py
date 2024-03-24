class Pathes():
    def __init__(self):
        pathToImages = '../../Img/'
        sourceFileName = 'leopard.jpg'#'leopardSmall.bmp'
        destinyFileName = 'leopardOut.bmp'
        self._sourcePath = pathToImages + sourceFileName
        self._destinyPath = pathToImages + destinyFileName
        

    @property
    def source(self):
        return self._sourcePath
    
    @property
    def destiny(self):
        return self._destinyPath

path = Pathes()

