from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification.Normal_distribution import Circuit

from FisherIris import DataPrep
# from FisherIris import VisualizeData
import FisherIris.fisher

def run_binary() -> None:
    data = create_data_set(100, 1.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Circuit.Model()
    model.train(lr=1.05, steps=30, trainX=data.trainX, trainY=data.trainY, sq=0.175)
    model.score_model(testX=data.testX, testY=data.testY)
    print('accuracy on train: ')
    model.predict(data_to_predict=data.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=data.testX)

    draw_decision_boundary(model=model, N_gridpoints=20, arrayX=data.trainX, arrayY=data.trainY)

def run_fisher() -> None:
    dataset = DataPrep.collect_fisher_dataset()
    model = FisherIris.fisher.Model()

    model.train(lr=.55, steps=250, trainX=dataset.trainX, trainY=dataset.trainY, sq=0.175)
    model.score_model(testX=dataset.testX, testY=dataset.testY)

    print('accuracy on train: ')
    model.predict(data_to_predict=dataset.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=dataset.testX)


if __name__ == '__main__':

    run_binary()
