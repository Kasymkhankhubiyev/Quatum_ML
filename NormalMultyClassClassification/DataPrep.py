import numpy as np
from typing import NamedTuple

class DataSet(NamedTuple):
    trainX: list
    trainY: list
    testX: list
    testY: list

def _create_normal_distributed_data(classes_scale: int, intersect_rate: float) -> DataSet:
    np.random.seed(0)
    l = classes_scale
    n = 2
    drop = intersect_rate

    X1 = np.array([[-2, -2]]) + drop * np.random.randn(l, n)
    X2 = np.array([[-2, 2]]) + drop * np.random.randn(l, n)
    X3 = np.array([[2, -2]]) + drop * np.random.randn(l, n)
    X4 = np.array([[2, 2]]) + drop * np.random.randn(l, n)

    # конкатенируем все в одну матрицу
    # при этом по 20 точек оставим на тест/валидацию
    X = np.vstack((X1[10:], X2[10:], X3[10:], X4[10:]))
    ValX = np.vstack((X1[:10], X2[:10], X3[:10], X4[:10]))

    # конкатенируем все в один столбец с соответствующими значениями для класса 0 или 1
    y = np.hstack([[0] * (l - 10), [1] * (l - 10), [2] * (l - 10), [3] * (l - 10)])
    ValY = np.hstack([[0] * 10, [1] * 10, [2] * 10, [3] * 10])

    return DataSet(trainX=X, trainY=y, testX=ValX, testY=ValY)


def create_data_set(classes_scale: int, intersect_rate: float) -> DataSet:
    # из этого можно сделать некоторую общую библиотеку
    return _create_normal_distributed_data(classes_scale, intersect_rate)