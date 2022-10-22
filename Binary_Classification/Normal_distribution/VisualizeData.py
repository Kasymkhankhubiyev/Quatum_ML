import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np


def visualize_dataset(trainX, trainY, testX, testY, name: str)-> None:
    colors = ['blue', 'red', 'yellow', 'green']

    for k in np.unique(trainY):
        plt.plot(trainX[trainY == k, 0], trainX[trainY == k, 1], 'o', label='class {}'.format(k), color=colors[k])

    for k in np.unique(testY):
        plt.plot(testX[testY == k, 0], testX[testY == k, 1], 'o', label='class {}'.format(k+2), color=colors[k+2])

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.savefig(name+'_classification_data_set.png')
    plt.close()


def visualize_data(arrayX, arrayY, name: str) -> None:
    pass