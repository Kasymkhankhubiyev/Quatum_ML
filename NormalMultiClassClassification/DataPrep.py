import numpy as np
from typing import NamedTuple


class DataSet(NamedTuple):
    trainX_0: list
    trainX_1: list
    trainX_2: list
    trainX_3: list
    trainY_0: list
    trainY_1: list
    trainY_2: list
    trainY_3: list
    testX_0: list
    testX_1: list
    testX_2: list
    testX_3: list
    testY_0: list
    testY_1: list
    testY_2: list
    testY_3: list

def _create_normal_distributed_data(classes_scale: int, intersect_rate: float) -> DataSet:
    np.random.seed(0)
    l = classes_scale
    n = 2
    drop = intersect_rate

    X0 = np.array([[-2, -2]]) + drop * np.random.randn(l, n)
    X1 = np.array([[-2, 2]]) + drop * np.random.randn(l, n)
    X2 = np.array([[2, -2]]) + drop * np.random.randn(l, n)
    X3 = np.array([[2, 2]]) + drop * np.random.randn(l, n)

    dataset = DataSet(
        trainX_0=X0[10:], trainY_0=[0] * (l - 10), testX_0=X0[:10], testY_0=[0] * 10,
        trainX_1=X1[10:], trainY_1=[1] * (l - 10), testX_1=X1[:10], testY_1=[1] * 10,
        trainX_2=X2[10:], trainY_2=[2] * (l - 10), testX_2=X2[:10], testY_2=[2] * 10,
        trainX_3=X3[10:], trainY_3=[3] * (l - 10), testX_3=X3[:10], testY_3=[3] * 10)

    return dataset


def create_data_set(classes_scale: int, intersect_rate: float) -> DataSet:
    # из этого можно сделать некоторую общую библиотеку
    return _create_normal_distributed_data(classes_scale, intersect_rate)