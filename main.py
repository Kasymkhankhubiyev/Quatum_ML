from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification import Normal_distribution

from FisherIris import DataPrep
# from FisherIris import VisualizeData
import FisherIris.fisher

def run_binary() -> None:
    data = create_data_set(100, 1.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Normal_distribution.Circuit.Model()
    model.train(lr=1.05, steps=20, trainX=data.trainX, trainY=data.trainY, sq=0.175)
    model.score_model(testX=data.testX, testY=data.testY)
    print('accuracy on train: ')
    model.predict(data_to_predict=data.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=data.testX)


if __name__ == '__main__':

    dataset = DataPrep.collect_fisher_dataset()
    model = FisherIris.fisher.Model()

    model.train(lr=.5, steps=200, trainX=dataset.trainX, trainY=dataset.trainY, sq=0.175)
    model.score_model(testX=dataset.testX, testY=dataset.testY)

    print('accuracy on train: ')
    model.predict(data_to_predict=dataset.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=dataset.testX)
