from NormalMultyClassClassification.DataPrep import *
from NormalMultyClassClassification.VisualizeData import *
from NormalMultyClassClassification.Model import Model

def run() -> None:
    data = create_data_set(100, .5)
    visualize_dataset(data.trainX, data.trainY, data.testX, data.testY, 'Normal 0.5')
    model = Model()
