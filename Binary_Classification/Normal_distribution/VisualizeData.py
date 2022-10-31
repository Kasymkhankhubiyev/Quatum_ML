import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import numpy as np
# from Circuit import Model


def visualize_dataset(trainX, trainY, testX, testY, name: str) -> None:
    colors = ['blue', 'red', 'yellow', 'green']

    for k in np.unique(trainY):
        plt.plot(trainX[trainY == k, 0], trainX[trainY == k, 1], 'o', label='class {}'.format(k), color=colors[k])

    for k in np.unique(testY):
        plt.plot(testX[testY == k, 0], testX[testY == k, 1], 'o', label='class {}'.format(k+2), color=colors[k+2])

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/'+name+'_classification_data_set.png')
    plt.close()


def visualize_data(arrayX, arrayY, name: str) -> None:
    colors = ['blue', 'red', 'yellow', 'green']
    for k in np.unique(arrayY):
        plt.plot(arrayX[arrayY == k, 0], arrayX[arrayY == k, 1], 'o', label='class {}'.format(k), color=colors[k])
    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/' + name + '_transformed_data_set.png')
    plt.close()


def draw_decision_boundary(model, arrayX, arrayY, name: str, N_gridpoints=14) -> None:

    #разбиваем область на точки
    _xx, _yy = np.linspace(-4, 4, N_gridpoints), np.linspace(-4, 4, N_gridpoints)

    _zz = np.zeros_like(_xx)

    # points = np.array([[_xx[i], _yy[i]] for i in range(len(_xx))])

    points = []
    for i in range(N_gridpoints):
        for j in range(N_gridpoints):
            points.append([_xx[i], _yy[j]])

    _zz = model.predict(points)

    _zz = [_zz[i] for i in range(len(_zz))]

    colors = ['pink', 'purple']

    # for k in np.unique(_zz):
    #     plt.scatter(x=points[_zz == k][0], y=points[_zz == k][1], s=30, c=colors[k])
    for i in range(len(points)):
        if _zz[i] == 0:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[0], marker='s')
        elif _zz[i] == 1:
            plt.scatter(x=points[i][0], y=points[i][1], s=150, c=colors[1], marker='s')

    colors = ['red', 'yellow']

    # for i in range(len(arrayX)):
    #     if arrayY[i] == 1:
    #         plt.plot(arrayX[i][0], arrayX[i][1], 'o', color=colors[0])
    #     elif arrayY[i] == 0:
    #         plt.plot(arrayX[i][0], arrayX[i][1], 'o', color=colors[1])
    for k in np.unique(arrayY):
        plt.plot(arrayX[arrayY == k, 0], arrayX[arrayY == k, 1], 'o', label='class {}'.format(k), color=colors[k])

    # name = 'decision_boundary_test_plot'

    plt.legend(fontsize=7, ncol=1, facecolor='oldlace', edgecolor='r')
    plt.title(name)
    plt.savefig('Binary_Classification/Normal_distribution/' + name + '_decision_boundary.png')
    plt.close()
