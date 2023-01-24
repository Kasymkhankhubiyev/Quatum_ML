from Mnist.CNN.DataPrep import create_dataset, create_dataset_binary
from Mnist.CNN.VisualizeData import draw_samples
from Mnist.CNN.Model import Model
from Mnist.QCNN.Circuit import CNN
import numpy as np


def run() -> None:
    digits = create_dataset_binary(class0=1, class1=7)
    # draw_samples(dataset=digits, samples_number=8)
    print(digits.testX.shape)
    print(digits.testY)
    # train_arr = []
    # for i in range(5):
    #     train_arr.append([i]*16)
    # gt_arr = [1, 1, 0, 0, 1]
    # train_arr = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
    model = Model()
    # model = CNN()
    # model.fit(lr=1.75, sq=1.5, steps=5, trainX=np.array(train_arr), trainY=np.array(gt_arr))
    print(f'trainX = {digits.trainX.shape}')
    model.fit(lr=1.75, sq=1.5, steps=5, clf_task='binary', trainX=digits.trainX[:5], trainY=digits.trainY[:5])
    # model.fit(lr=1.75, sq=1.5, steps=5, clf_task='binary', trainX=np.array(train_arr), trainY=np.array(gt_arr))
    model.predict(data_to_predict=digits.testX)
    model.score_model(testX=digits.testX, testY=digits.testY)
