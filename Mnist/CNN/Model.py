import strawberryfields as sf
import numpy as np

from exceptions import DoesntMatchChosenTask


class Model:
    """
    cf_tasl: classification type, if 'binary' - binary classification,
    'multi' - multiclass classification
    """

    def __init__(self, clf_task: str, params=None,) -> None:
        self.params, self.squeeze_rate = None, None
        self.learner, self.clf_task = None, clf_task
        if clf_task not in ['binary', 'multi']:
            raise DoesntMatchChosenTask(tasks_list=['binary', 'multi'], err_task=clf_task)

    def _layer(self):
        pass

    def _circuit(self):
        pass

    def predict(self) -> np.array:
        pass

    def fit(self) -> None:
        pass
