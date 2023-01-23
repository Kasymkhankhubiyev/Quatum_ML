"""
most common and general functions helping to build a model
"""

import numpy as np
from qmlt.numerical import CircuitLearner
import datetime


def _outputs_to_predictions(outputs: np.array) -> np.array:
    pass


def predict(self, learner: CircuitLearner, data_to_predict: np.array) -> np.array:
    outcomes = learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
    predictions = outcomes['predictions']
    return predictions


def upload_params(file_name: str, data: list, input_name='params') -> None:
    name = file_name
    with open(name, 'a') as file:
        file.write(input_name+'/_on_'+datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d")+'\n\n')
        for i in range(len(data)):
            file.write(str(data[i])+',')
        file.write('\n\n\n')
