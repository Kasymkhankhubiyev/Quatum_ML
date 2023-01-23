import strawberryfields as sf
import numpy as np
import datetime

from exceptions import DoesntMatchChosenTask
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss


class MainModel:
    def __init__(self, params=None) -> None:
        self.params, self.squeeze_rate = None, None
        self.learner, self.clf_task = None, None
        self.lr, self.steps = None, None
        self.X, self.Y = None, None
        if params is not None:
            pass

    def _outputs_to_predictions(self, circuit_output):
        return np.round(circuit_output)

    def predict(self, data_to_predict) -> np.array:
        outcomes = self.learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions
