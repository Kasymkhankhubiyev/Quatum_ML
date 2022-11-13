from NormalMultyClassClassification.DataPrep import *
from NormalMultyClassClassification.VisualizeData import *
from NormalMultyClassClassification.Model import Model

def run() -> None:
    data = create_data_set(100, .5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    trainX = np.array([[data.trainX[i][0], data.trainX[i][1], 0, 0] for i in range(len(data.trainX))])
    testX = np.array([[data.testX[i][0], data.testX[i][1], 0, 0] for i in range(len(data.testX))])
    # print(trainX)
    model = Model()
    model.train(lr=0.5, steps=10, sq=0.175, trainX=trainX, trainY=data.trainY)
    model.score_model(testX=testX, testY=data.testY)
