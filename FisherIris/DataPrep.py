from sklearn import datasets
import numpy as np
from typing import NamedTuple


class Dataset(NamedTuple):
    trainX: np.array
    trainY: np.array
    testX: np.array
    testY: np.array


def collect_fisher_dataset() -> Dataset:

    iris = datasets.load_iris()
    trainX = iris.data
    trainY = iris.target

    testX = []
    testY = []

    # забираем 15 штук на валидацию, выбрал рандомно
    indexes = [42, 25, 98, 50, 136, 113, 90, 118, 7, 81, 114, 128, 46, 103, 63]

    for index in indexes:
        testX.append(trainX[index])
        testY.append(trainY[index])

    trainX = np.delete(trainX, indexes, axis=0)
    trainY = np.delete(trainY, indexes, axis=0)

    return Dataset(trainX=np.array(trainX), trainY=np.array(trainY),
                   testX=np.array(testX), testY=np.array(testY))


