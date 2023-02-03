from Mnist.CNN.DataPrep import create_dataset, create_dataset_binary
from Mnist.CNN.VisualizeData import draw_samples
from Mnist.CNN.Model import Model


def run() -> None:
    digits = create_dataset_binary(class0=1, class1=4)
    # draw_samples(dataset=digits, samples_number=8)
    print(digits.testX.shape)
    print(digits.testY)
    model = Model()
    print(f'trainX = {digits.trainX.shape}')
    model.fit(lr=.5, sq=1.5, steps=20, clf_task='binary', train_x=digits.trainX, train_y=digits.trainY)
    model.predict(data_to_predict=digits.testX)
    model.score_model(test_x=digits.testX, test_y=digits.testY)
