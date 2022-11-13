from NormalMultyClassClassification.DataPrep import *
from NormalMultyClassClassification.VisualizeData import *
# import NormalMultyClassClassification as nm
import NormalMultyClassClassification.ModelOnevAll as ova

def run() -> None:
    data = create_data_set(50, .5)
    # visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    # trainX = np.array([[data.trainX[i][0], data.trainX[i][1], 0, 0] for i in range(len(data.trainX))])
    # testX = np.array([[data.testX[i][0], data.testX[i][1], 0, 0] for i in range(len(data.testX))])
    # print(trainX)
    # model = nm.Model.Model()
    model = ova.Model()
    model.train(lr=0.5, steps=10, sq=0.175, dataset=data)
    test_x = np.vstack((data.testX_0, data.testX_1, data.testX_2, data.testX_3))
    test_y = np.hstack([data.testY_0, data.testY_1, data.testY_2, data.testY_3])
    prediction = model.predict(test_x)
    print(prediction)
    # model.score_model(testX=test_x, testY=test_y)
