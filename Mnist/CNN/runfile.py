from Mnist.CNN.DataPrep import create_dataset_binary
from Mnist.CNN.Models.SimpleModel import Model


def run() -> None:
    digits = create_dataset_binary(class0=1, class1=4)
    model = Model()
    print(f'trainX = {digits.trainX.shape}')
    model.fit(lr=.175, sq=1.5, steps=15, clf_task='binary', train_x=digits.trainX, train_y=digits.trainY)
    model.predict(data_to_predict=digits.testX)
    model.score_model(test_x=digits.testX, test_y=digits.testY)
