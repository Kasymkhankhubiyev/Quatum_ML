# from Mnist.CNN.DataPrep import create_dataset_binary
# from Mnist.CNN.Models import SimpleModel
# from Mnist.CNN.Models import ComplexModel

from FisherIris.Model import Model
from FisherIris.DataPrep import collect_fisher_dataset, Dataset
import matplotlib.pyplot as plt


def run() -> None:
    # digits = create_dataset_binary(class0=1, class1=9)
    # # model = SimpleModel.Model()
    # model = ComplexModel.Model()
    # print(f'trainX = {digits.trainX.shape}')
    # model.fit(lr=.35, sq=1.5, steps=10, clf_task='binary', train_x=digits.trainX, train_y=digits.trainY)
    # model.predict(data_to_predict=digits.testX)
    # model.score_model(test_x=digits.testX, test_y=digits.testY)

    dataset = collect_fisher_dataset()

    test_acc, test_loss = [], []

    for step in range(1, 32, 10):
        model = Model()
        model.train(lr=0.5, sq=1.5, trainX=dataset.trainX, trainY=dataset.trainY, steps=step)
        acc, loss =  model.score_model(testX=dataset.testX, testY=dataset.testY, save=False)
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


