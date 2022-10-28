"""
Multiclass classification with the FisherIris dataset
"""
import strawberryfields as sf
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import cross_entropy_with_softmax
import matplotlib.pyplot as plt
import numpy as np
import datetime

squeeze_rate = float
learning_rate = float

class Model:

    def __init__(self) -> None:
        self.lr, self.steps, self.squeeze_param, self.learner = None, None, None, None
        self.params = [make_param(name = 'param' + str(i), constant=.5) for i in range(46)]

    def predict(self) -> None:
        pass

    def _circuit(self, X, params):
        # prog = sf.Program(2)

        def single_input_circuit(x):
            modes = 4
            prog = sf.Program(modes)
            squeezing_amount = self.squeeze_param

            with prog.context as q:
                # ops.Dgate(x[0], 0.) | q[0]
                # ops.Dgate(x[1], 0.) | q[1]
                # ops.Dgate(x[2], 0.) | q[2]
                # ops.Dgate(x[3], 0.) | q[3]
                ops.Sgate(squeezing_amount, x[0]) | q[0]
                ops.Sgate(squeezing_amount, x[1]) | q[1]
                ops.Sgate(squeezing_amount, x[2]) | q[2]
                ops.Sgate(squeezing_amount, x[3]) | q[3]
                ops.BSgate(self.params[0], self.params[1]) | (q[0], q[1])
                ops.BSgate(self.params[2], self.params[3]) | (q[2], q[3])
                ops.BSgate(self.params[4], self.params[5]) | (q[1], q[2])
                ops.BSgate(self.params[6], self.params[7]) | (q[0], q[1])
                ops.BSgate(self.params[9], self.params[8]) | (q[2], q[3])
                ops.BSgate(self.params[10], self.params[11]) | (q[1], q[2])
                ops.Rgate(self.params[12]) | q[0]
                ops.Rgate(self.params[13]) | q[1]
                ops.Rgate(self.params[14]) | q[2]
                ops.Sgate(self.params[15]) | q[0]
                ops.Sgate(self.params[16]) | q[1]
                ops.Sgate(self.params[17]) | q[2]
                ops.Sgate(self.params[18]) | q[3]
                ops.BSgate(self.params[19], self.params[20]) | (q[0], q[1])
                ops.BSgate(self.params[21], self.params[22]) | (q[2], q[3])
                ops.BSgate(self.params[23], self.params[24]) | (q[1], q[2])
                ops.BSgate(self.params[25], self.params[26]) | (q[0], q[1])
                ops.BSgate(self.params[27], self.params[28]) | (q[2], q[3])
                ops.BSgate(self.params[29], self.params[30]) | (q[1], q[2])
                ops.Rgate(self.params[31]) | q[0]
                ops.Rgate(self.params[32]) | q[1]
                ops.Rgate(self.params[33]) | q[2]
                ops.Dgate(self.params[34]) | q[0]
                ops.Dgate(self.params[35]) | q[1]
                ops.Dgate(self.params[36]) | q[2]
                ops.Dgate(self.params[37]) | q[3]
                ops.Pgate(self.params[38]) | q[0]
                ops.Pgate(self.params[39]) | q[1]
                ops.Pgate(self.params[40]) | q[2]
                ops.Pgate(self.params[41]) | q[3]
                ops.Kgate(self.params[42]) | q[0]
                ops.Kgate(self.params[43]) | q[1]
                ops.Kgate(self.params[44]) | q[2]
                ops.Kgate(self.params[45]) | q[3]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})

            result = eng.run(prog)
            state = result.state

            ei = [0, 0, 0, 0]  # modes*[0]
            ei[0] = 2
            p0 = state.fock_prob(ei)
            ei[0] = 0
            ei[1] = 2
            p1 = state.fock_prob(ei)
            ei[1] = 0
            ei[2] = 2
            p2 = state.fock_prob(ei)
            ei[2] = 0

            # print('probobilitis:      ')
            # print(p0, p1, p2)

            normalization = p0 + p1 + p2 + 1e-10
            output = [p0 / normalization, p1 / normalization, p2 / normalization]
            # print('predictions:   ')
            # print(output)

            return output

        circuit_output = [single_input_circuit(x) for x in X]

        return np.array(circuit_output)

    def _myloss(self, circuit_output, targets):
        # return tf.losses.mean_squared_error(y_pred=circuit_output, y_true=targets)
        # print(circuit_output)
        # print(targets)
        return cross_entropy_with_softmax(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        for i in range(len(circuit_output)):
            circuit_output[i] = round(circuit_output[i])
        return circuit_output

    def _upload_params(self):
        name = 'FisherIris/params_on_'+(datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_M"))+'.txt'
        with open(name, 'w') as file:
            for i in range(len(self.params)):
                file.write(str(self.params[i])+',')

    def train(self, lr: learning_rate, sq: squeeze_rate, steps: int, trainX: np.array, trainY: np.array) -> None:

        self.lr = lr
        self.squeeze_param = sq
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': self.lr,
                       # 'decay': 0.01,
                       'log_every': 1,
                       'warm_start': False
                       }

        self.learner = CircuitLearner(hyperparams=hyperparams)

        self.learner.train_circuit(X=trainX, Y=trainY, steps=steps)
