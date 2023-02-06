from .DataPrep import  collect_fisher_dataset, collect_data_fisher_for_decision_boundary
from .Model import Model
from .ParamsParcers import parce_params
from .VisualizeData import draw_decision_boundary


if __name__ == '__main__':
    dataset = collect_fisher_dataset()
    model = Model()

    model.train(lr=.55, steps=250, trainX=dataset.trainX, trainY=dataset.trainY, sq=1.575)
    model.score_model(testX=dataset.testX, testY=dataset.testY)

    print('accuracy on train: ')
    model.predict(data_to_predict=dataset.trainX)
    print('accuracy on test: ')
    model.predict(data_to_predict=dataset.testX)

    # params = parce_params(file_name='FisherIris/params_on_2022_10_29_22_M.txt')
    # trainX, trainY = collect_data_fisher_for_decision_boundary()
    # model = Model(params=params)
    #
    # draw_decision_boundary(model=model, arrX=trainX, arrY=trainY, N_gridpoints=10, name='Fisher')
