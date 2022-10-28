"""
Multiclass classification with the FisherIris dataset
"""
import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import cross_entropy_with_softmax
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import pandas as pd
# import tensorflow as tf

class Model:

    def __init__(self) -> None:
        self.lr, self.steps = None, None
        self.params = None

    def predict(self) -> None:
        pass

    def train(self) -> None:
        pass
