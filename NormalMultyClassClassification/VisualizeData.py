import matplotlib.pyplot as plt
import numpy as np

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