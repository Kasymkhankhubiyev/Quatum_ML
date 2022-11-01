import matplotlib.pyplot as plt
import numpy as np
# from FisherIris.DataPrep import Dataset


def visualize_dataset(dataset) -> None:
    fig, axs = plt.subplots(nrows=4, ncols=4)

    colors = ['red', 'blue', 'green']
    test_colors = ['purple', 'black', 'yellow']

    for i in range(4):
        for j in range(4):
            for k in np.unique(dataset.trainY):
                axs[i % 4, j % 4].plot(dataset.trainX[dataset.trainY == k, i],
                                       dataset.trainX[dataset.trainY == k, j],
                                       'o', label='class {}'.format(k), color=colors[k]
                                       )

                axs[i % 4, j % 4].plot(dataset.testX[dataset.testY == k, i],
                                       dataset.testX[dataset.testY == k, j],
                                       'o', label='test_class {}'.format(k), color=test_colors[k]
                                       )

                axs[i % 4, j % 4].legend(fontsize=5,
                                         ncol=1,
                                         facecolor='oldlace',
                                         edgecolor='r'
                                         )

                # axs[i % 4, j % 4].set_xlim(-1, 10)
                # axs[i % 4, j % 4].set_ylim(-1, 10)

    fig.set_size_inches(10., 6.5)
    plt.savefig('FisherIris/dataset_visualization.png')


def draw_decision_boundary(model, arrX, arrY, name: str, N_gridpoints=14) -> None:

    _x1 = np.linspace(arrX[:, 0].min() - 1, arrX[:, 0].max() + 1, N_gridpoints)
    _x2 = np.linspace(arrX[:, 1].min() - 1, arrX[:, 1].max() + 1, N_gridpoints)
    _x3 = np.linspace(arrX[:, 2].min() - 1, arrX[:, 2].max() + 1, N_gridpoints)
    _x4 = np.linspace(arrX[:, 3].min() - 1, arrX[:, 3].max() + 1, N_gridpoints)

    points = [_x1, _x2, _x3, _x4]
    # fig, axs = plt.subplots(nrows=4, ncols=4)

    predictions = []
    for i in range(N_gridpoints):
        for j in range(N_gridpoints):
            for k in range(N_gridpoints):
                for m in range(N_gridpoints):
                    predictions.append([_x1[i], _x2[j], _x3[k], _x4[m]])

    _zz = model.predict(predictions)

    _zz = [_zz[i] for i in range(len(_zz))]  # np.where(_zz[i] == 1)
    for i in range(len(_zz)):
        index, = np.where(_zz[i] == 1)
        _zz[i] = index

    colors = ['red', 'blue', 'green']
    test_colors = ['pink', 'SteelBlue', 'yellow']

    for i in range(4):
        for j in range(4):
            fig, axs = plt.subplots(nrows=1, ncols=2)
            for k in range(len(_zz)):
                if _zz[k] == 0:
                    axs[0].scatter(x=predictions[k][i], y=predictions[k][j], s=130, c=test_colors[0], marker='s')
                elif _zz[k] == 1:
                    axs[0].scatter(x=predictions[k][i], y=predictions[k][j], s=130, c=test_colors[1], marker='s')
                elif _zz[k] == 2:
                    axs[0].scatter(x=predictions[k][i], y=predictions[k][j], s=130, c=test_colors[2], marker='s')

            for k in np.unique(arrY):
                axs[1].plot(arrX[arrY == k, i],
                                       arrX[arrY == k, j],
                                       'o', label='class {}'.format(k), color=colors[k]
                                       )

                axs[1].legend(fontsize=5,
                                         ncol=1,
                                         facecolor='oldlace',
                                         edgecolor='r'
                                         )

                # index, = np.where(arrY[k] == 1)
                # if index == 0:
                #     axs[1].plot(arrX[k][i], arrX[k][j], color=colors[0])
                # elif index == 1:
                #     axs[1].plot(arrX[k][i], arrX[k][j], color=colors[1])
                # elif index == 2:
                #     axs[1].plot(arrX[k][i], arrX[k][j], color=colors[2])

            plt.title(f'f{i} vs f{j}')
            plt.savefig('FisherIris/decision_boundary_' + str(i) + 'vs' + str(j) + '_' + str(N_gridpoints) + '.png')
            plt.close(fig)

    # for i in range(4):
    #     for j in range(4):
    #         for k in np.unique(arrY):
    #             index, = np.where(arrY[k] == 1)
    #             if index == 0:
    #                 axs[i % 4, j % 4].plot(arrX[k][i], arrX[k][j], color=colors[0])
    #             elif index == 1:
    #                 axs[i % 4, j % 4].plot(arrX[k][i], arrX[k][j], color=colors[1])
    #             elif index == 2:
    #                 axs[i % 4, j % 4].plot(arrX[k][i], arrX[k][j], color=colors[2])

    # fig.set_size_inches(10., 6.5)
    # plt.savefig('FisherIris/decision_boundary'+str(N_gridpoints)+'.png')


