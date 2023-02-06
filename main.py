from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification.Normal_distribution import Circuit
# from NormalMultiClassClassification import runfile
# from FisherIris import DataPrep
# from FisherIris import VisualizeData
# import FisherIris.Model
# from FisherIris import ParamsParcers
# from FisherIris import VisualizeData
from Mnist.CNN.runfile import run
from sklearn.datasets import load_digits


def run_binary() -> None:
    data = create_data_set(20, 1.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Circuit.Model()
    model.train(lr=0.75, steps=50, trainX=data.trainX, trainY=data.trainY, sq=1.575)
    model.score_model(testX=data.testX, testY=data.testY)
    print('accuracy on train: ')
    model.predict(data_to_predict=data.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=data.testX)

def run_binary_complex() -> None:
    data = create_symmetric_dataset(100, 0.5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Circuit.Model()
    model.train(lr=0.75, steps=100, trainX=data.trainX, trainY=data.trainY, sq=1.575)
    model.score_model(testX=data.testX, testY=data.testY)
    print('accuracy on train: ')
    model.predict(data_to_predict=data.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=data.testX)

    draw_decision_boundary(model=model, N_gridpoints=20, arrayX=data.trainX, arrayY=data.trainY, name='symmetric')


# def run_fisher_decision_bound() -> None:
#     params = ParamsParcers.parce_params(file_name='FisherIris/params_on_2022_10_29_22_M.txt')
#     trainX, trainY = DataPrep.collect_data_fisher_for_decision_boundary()
#     model = FisherIris.fisher.Model(params=params)
#
#     VisualizeData.draw_decision_boundary(model=model, arrX=trainX, arrY=trainY, N_gridpoints=10, name='Fisher')


if __name__ == '__main__':

    # runfile.run()
    # run_fisher()
    run()
    # run_binary_complex()
    # run_binary()
    # data = load_digits()
    # x, y = np.array(data.data), np.array(data.target)
    # trainX, trainY = x[np.where(y==3)], y[np.where(y==3)]
    #
    #
    # print(y[np.where(y==3)])


