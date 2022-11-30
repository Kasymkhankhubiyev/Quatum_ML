from sklearn.datasets import load_digits
from typing import NamedTuple

import numpy as np


class Dataset(NamedTuple):
    trainX: np.array
    trainY: np.array
    testX: np.array
    testY: np.array


def create_dataset() -> Dataset:
    """
    Data load from sklearn datasets. All data and targets are mixed.
    Test is 10% of the dataset.
    :return: Dataset of digits divided into train ans test subsets
    """
    # загружаем датасет
    digits = load_digits()
    # (trainX, trainY), (testX, testY) =

    x = np.array(digits.data)
    y = np.array(digits.target)
    trainY, trainX = [], []

    # mix
    indexes = np.random.permutation(len(y))
    for i in range(len(y)):
        trainY.append(y[indexes[i]])
        trainX.append(x[indexes[i]])

    sep = round(len(trainY)*0.1) # ~10% for a test

    return Dataset(testX=np.array(trainX[sep:]), trainX=np.array(trainX[:sep]),
                   testY=np.array(trainY[sep:]), trainY=np.array(trainY[:sep]))


    #Изменяем размер изображения в пикселях
    # trainX = trainX.reshape((trainX.shape[0], 16, 16, 1))
    # testX = trainX.reshape((trainX.shape[0], 16, 16, 1))

    # переводим вектор в матричный вид, т.е. 1 там, где правильный класс,
    # в других позициях 0 --- Нужно ли?
