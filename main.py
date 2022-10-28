from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification.Normal_distribution.Circuit import Model

from FisherIris import DataPrep
from FisherIris import VisualizeData

def run_binary() -> None:
    data = create_data_set(100, 1.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Model()
    model.train(lr=1.05, steps=100, trainX=data.trainX, trainY=data.trainY, sq=0.175)
    model.score_model(testX=data.testX, testY=data.testY)
    print('accuracy on train: ')
    model.predict(data_to_predict=data.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=data.testX)
    # model.project_data_onto_CV(arrX=data.trainX, arrY=data.trainY)
    # print(data.trainX)
    # print(model.transform_data)
    # print(model.params[0]['val'])


if __name__ == '__main__':

    dataset = DataPrep.collect_fisher_dataset()


