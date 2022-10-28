import matplotlib.pyplot as plt
import numpy as np
from FisherIris.DataPrep import Dataset


def visualize_dataset(dataset: Dataset) -> None:
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

                axs[i % 4, j % 4].legend(fontsize=7,
                                         ncol=1,
                                         facecolor='oldlace',
                                         edgecolor='r'
                                         )

    fig.set_size_inches(18.5, 10.5)
    plt.savefig('FisherIris/dataset_visualization.png')

    def visualize_transformed_data(data):
        pass
