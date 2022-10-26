from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification.Normal_distribution.Circuit import Model
# import strawberryfields as sf

if __name__ == '__main__':

    data = create_data_set(100, 1.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Model()
    model.train(lr=0.5, steps=20, trainX=data.trainX, trainY=data.trainY, sq=0.195)
    model.score_model(testX=data.testX, testY=data.testY)
    model.predict(data_to_predict=data.testX)
    model.project_data_onto_CV(arrX=data.trainX, arrY=data.trainY)
    print(data.trainX)
    # print(model.transform_data)
    print(model.params[0]['val'])
