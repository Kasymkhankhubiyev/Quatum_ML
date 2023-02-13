from Mnist.CNN.DataPrep import create_dataset_binary
from Mnist.CNN.Models import SimpleModel
from Mnist.CNN.Models import ComplexModel


def run() -> None:
    digits = create_dataset_binary(class0=1, class1=9)
    # model = SimpleModel.Model()
    model = ComplexModel.Model()
    print(f'trainX = {digits.trainX.shape}')
    model.fit(lr=.35, sq=1.5, steps=10, clf_task='binary', train_x=digits.trainX, train_y=digits.trainY)
    model.predict(data_to_predict=digits.testX)
    model.score_model(test_x=digits.testX, test_y=digits.testY)
