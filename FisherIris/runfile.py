from .DataPrep import collect_fisher_dataset
from .Model import Model
# from .ParamsParcers import parce_params
# from .VisualizeData import draw_decision_boundary
import matplotlib.pyplot as plt


def run():
    # dataset = collect_fisher_dataset()
    # model = Model()
    #
    # model.train(lr=.55, steps=250, trainX=dataset.trainX, trainY=dataset.trainY, sq=1.575)
    # model.score_model(testX=dataset.testX, testY=dataset.testY)
    #
    # print('accuracy on train: ')
    # model.predict(data_to_predict=dataset.trainX)
    # print('accuracy on test: ')
    # model.predict(data_to_predict=dataset.testX)

    # params = parce_params(file_name='FisherIris/params_on_2022_10_29_22_M.txt')
    # trainX, trainY = collect_data_fisher_for_decision_boundary()
    # model = Model(params=params)
    #
    # draw_decision_boundary(model=model, arrX=trainX, arrY=trainY, N_gridpoints=10, name='Fisher')

    dataset = collect_fisher_dataset()

    test_acc, test_loss = [], []

    for step in range(20, 132, 10):
        model = Model()
        model.train(lr=0.5, sq=1.5, trainX=dataset.trainX, trainY=dataset.trainY, steps=step)
        acc, loss = model.score_model(testX=dataset.testX, testY=dataset.testY, save=False)
        test_acc.append(acc)
        test_loss.append(loss)

        print(f'accuracy: {test_acc}')
        print(f'loss: {test_loss}')

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(range(1, 32, 10), test_acc, label='accuracy')
    ax[0].legend()

    ax[1].plot(range(1, 32, 10), test_loss, label='loss')
    ax[1].legend()

    plt.savefig()
