from Binary_Classification.Normal_distribution.DataPreparation import *
from Binary_Classification.Normal_distribution.VisualizeData import *
from Binary_Classification.Normal_distribution import Circuit
# from NormalMultiClassClassification import runfile
from Mnist.CNN.runfile import run


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


if __name__ == '__main__':
    run()
    # run_binary_complex()
    # run_binary()
