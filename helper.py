"""
most common and general functions helping to build a model
"""

import numpy as np
from qmlt.numerical import CircuitLearner

def _outputs_to_predictions(outputs: np.array) -> np.array:
    pass

def predict(self, learner: CircuitLearner, data_to_predict: np.array) -> np.array:
    outcomes = learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
    predictions = outcomes['predictions']
    return predictions
