from Mnist.CNN.DataPrep import create_dataset
from Mnist.CNN.VisualizeData import draw_samples
from Mnist.CNN.Model import Model


def run() -> None:
    digits = create_dataset()
    print(digits.testX)
    draw_samples(dataset=digits, samples_number=8)

    model = Model()
    model.fit(lr=0.5, sq=1.5, steps=10, clf_task='binary', trainX=digits.trainX, trainY=digits.trainY)


