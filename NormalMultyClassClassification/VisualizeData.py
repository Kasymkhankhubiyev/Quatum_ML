import matplotlib.pyplot as plt
import numpy as np
from NormalMultyClassClassification.DataPrep import DataSet
from NormalMultyClassClassification.ModelOnevAll import Model

def visualize_dataset(trainX, trainY, testX, testY, name: str) -> None:
    colors = ['blue', 'red', 'yellow', 'green', 'black', 'pink', 'orange', 'purple']

    for k in np.unique(trainY):
        plt.plot(trainX[trainY == k, 0], trainX[trainY == k, 1], 'o', label='class {}'.format(k), color=colors[k])

    for k in np.unique(testY):
        plt.plot(testX[testY == k, 0], testX[testY == k, 1], 'o', label='class {}'.format(k), color=colors[k+4])

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('NormalMultyClassClassification/'+name+'_classification_data_set.png')
    plt.close()

def plot_decision_boudary(model: Model, dataset: DataSet, N_gridpoints: int, name: str) -> None:
    # разбиваем область на точки
    _xx, _yy = np.linspace(-4, 4, N_gridpoints), np.linspace(-4, 4, N_gridpoints)

    _zz = np.zeros_like(_xx)

    # points = np.array([[_xx[i], _yy[i]] for i in range(len(_xx))])

    points = []
    for i in range(N_gridpoints):
        for j in range(N_gridpoints):
            points.append([_xx[i], _yy[j]])

    _zz = model.predict(points)

    _zz = [_zz[i] for i in range(len(_zz))]

    colors = ['pink', 'purple', 'blue', 'red']

    for i in range(len(points)):
        if _zz[i] == 0:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[0], marker='s')
        elif _zz[i] == 1:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[1], marker='s')
        elif _zz[i] == 2:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[2], marker='s')
        elif _zz[i] == 3:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[3], marker='s')

    dataset_colors = ['green', 'yellow', 'black', 'orange']

    trainX = np.vstack((dataset.trainX_0, dataset.trainX_1, dataset.trainX_2, dataset.trainX_3))
    trainY = np.hstack((dataset.trainY_0, dataset.trainY_1, dataset.trainY_2, dataset.trainY_3))
    for k in np.unique(trainY):
        plt.plot(trainX[trainY == k, 0], trainX[trainY == k, 1], 'o', label=f'class {k} + {colors[k]}',
                 color=dataset_colors[k])

    # name = 'decision_boundary_test_plot'

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('NormalMultyClassClassification/NormalMulty' + name + '_decision_boundary.png')
    plt.close()
