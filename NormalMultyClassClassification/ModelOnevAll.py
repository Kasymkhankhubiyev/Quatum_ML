import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import cross_entropy_with_softmax
import datetime

squeeze_rate = float
learning_rate = float


class Model:
    def __init__(self) -> None:
        pass
