from Mnist.CNN.DataPrep import Dataset

import matplotlib.pyplot as plt
import numpy as np


def draw_samples(dataset: Dataset, samples_number: int) -> None:
    """
    Visualize an ordered amount of randomly chosen samples from a given dataset.
    :param dataset: A Dataset from which to take samples to visualize
    :param samples_number: A number of samples to visualize
    """

    if samples_number % 4 == 0:
        rows = samples_number//4
    else:
        rows = samples_number // 4 + 1

    if samples_number > 4:
        fig, axs = plt.subplots(nrows=rows, ncols=4)
        for i in range(samples_number):
            index = np.random.randint(len(dataset.trainY))
            axs[i // 4, i % 4].imshow(dataset.trainX[index, :].reshape([8, 8]))
    else:
        fig, axs = plt.subplots(nrows=rows, ncols=4)
        for i in range(samples_number):
            index = np.random.randint(len(dataset.trainY))
            axs[i % 4].imshow(dataset.trainX[index, :].reshape([8, 8]))

    plt.savefig('Mnist/CNN/Digits.png')
