from sklearn.datasets import load_digits
from typing import NamedTuple

import numpy as np

class Dataset(NamedTuple):
    trainX: np.array
    trainY: np.array
    testX: np.array
    testY: np.array


def create_dataset():
    # загружаем датасет
    (trainX, trainY), (testX, testY) = mnist.load_data()

    #Изменяем размер изображения в пикселях
    trainX = trainX.reshape((trainX.shape[0], 16, 16, 1))
    testX = trainX.reshape((trainX.shape[0], 16, 16, 1))

    # переводим вектор в матричный вид, т.е. 1 там, где правильный класс,
    # в других позициях 0 --- Нужно ли?
    trainY = to_categorical()
