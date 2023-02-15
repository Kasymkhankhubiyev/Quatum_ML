from sklearn.datasets import load_digits
from typing import NamedTuple

import numpy as np


class Dataset(NamedTuple):
    trainX: np.array
    trainY: np.array
    testX: np.array
    testY: np.array


def _mix_data(x, y):
    trainY, trainX = [], []

    # mix
    indexes = np.random.permutation(len(y))
    for i in range(len(y)):
        trainY.append(y[indexes[i]])
        trainX.append(x[indexes[i]])

    return trainX, trainY


def create_dataset() -> Dataset:
    """
    Data load from sklearn datasets. All data and targets are mixed.
    Test is 10% of the dataset.
    :return: Dataset of digits divided into train ans test subsets
    """
    # загружаем датасет
    digits = load_digits()

    x = np.array(digits.data) / 255.0
    y = np.array(digits.target)

    trainX, trainY = _mix_data(x, y)
    sep = round(len(trainY)*0.1)  # ~10% for a test

    return Dataset(testX=np.array(trainX[sep:]), trainX=np.array(trainX[:sep]),
                   testY=np.array(trainY[sep:]), trainY=np.array(trainY[:sep]))


def create_dataset_binary(class0: int, class1=None) -> Dataset:
    # загружаем датасет
    digits = load_digits()

    x = np.array(digits.data)
    y = np.array(digits.target)

    x0, y0 = x[y == class0], y[y == class0]
    y0 = np.zeros(len(y0))

    if class1 is not None:
        x1, y1 = x[y == class1], y[y == class1]
        y1 = np.ones(len(y1))
    else:
        x1, y1 = x[y != class0], y[y != class0]
        y1 = np.zeros(len(y1))

    # x, y = _mix_data(np.vstack((x0, x1)), np.hstack([y0, y1]))
    x, y = _mix_data(np.vstack((x0[:35], x1[:35])), np.hstack([y0[:35], y1[:35]]))

    sep = round(len(y) * 0.1)  # ~10% for a test

    return Dataset(trainX=np.array(x[sep:]) / 255.0, testX=np.array(x[:sep]) / 255.0,
                   trainY=np.array(y[sep:]), testY=np.array(y[:sep]))
