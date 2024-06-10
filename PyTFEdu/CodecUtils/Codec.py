from numpy._typing import NDArray
from CodecUtils.CompressedPhoto import CompressedPhoto
from CodecUtils.NeuroNet import NeuroNet
from CodecUtils.NeuroNetParameters import NeuroNetParameters
from CodecUtils.NpImageConverter import npImageConverter as ic
import numpy as np

class Codec():
    def __init__(self):
        None
        
    # Сценарий сжатия:
    # 1) Преобразовать картинку к последовательности векторов
    # 2) Скомпилировать всю нейронку
    # 3) Обучить всю нейронку
    # 4) Получить сжатые тайлы
    # 5) Уменьшить размерность данных тайлов
    # 6) Обучить декодер
    # 7) Получить веса
    # 8) Упаковать всё в объект
    # 9) Выдать объект
    def encode(self, npImage: NDArray, neuroNetParameters: NeuroNetParameters) -> CompressedPhoto:
        data, shape = ic.normalizeNpImage(npImage, neuroNetParameters.pixelsPerTile)
        neuroNet = NeuroNet()
        (_, _, chanels) = shape
        (compressed, restoreWeights) = neuroNet.encode(data, neuroNetParameters, chanels)
        compressedPhoto = CompressedPhoto(
            compressed, restoreWeights, neuroNetParameters.countRestoreLayers, shape, neuroNetParameters.pixelsPerTile)
        return compressedPhoto
        
    # Сценарий восстановления:
    # 1) Получить данные об архитектуре нейросети
    # 2) Восстановить нейросеть
    # 3) Разжать и выдать изображение
    def decode(self, compressedPhoto: CompressedPhoto) -> NDArray:
        neuroNet = NeuroNet()
        data = neuroNet.decode(compressedPhoto)
        restored = ic.recoverNpImage(data, compressedPhoto.shape)
        return restored



