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
        self.params = [make_param(name='param' + str(i), constant=.5) for i in range(46)]

    def predict(self, data_to_predict) -> list:
        outcomes = self.learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions

    def _circuit(self, X, params):

        def single_input_circuit(x):
            modes = 4
            prog = sf.Program(modes)
            squeezing_amount = self.squeeze_param

            with prog.context as q:
                ops.Sgate(squeezing_amount, x[0]) | q[0]
                ops.Sgate(squeezing_amount, x[1]) | q[1]
                ops.Sgate(squeezing_amount, x[2]) | q[2]
                ops.Sgate(squeezing_amount, x[3]) | q[3]
                ops.BSgate(params[0], params[1]) | (q[0], q[1])
                ops.BSgate(params[2], params[3]) | (q[2], q[3])
                ops.BSgate(params[4], params[5]) | (q[1], q[2])
                ops.BSgate(params[6], params[7]) | (q[0], q[1])
                ops.BSgate(params[9], params[8]) | (q[2], q[3])
                ops.BSgate(params[10], params[11]) | (q[1], q[2])
                ops.Rgate(params[12]) | q[0]
                ops.Rgate(params[13]) | q[1]
                ops.Rgate(params[14]) | q[2]
                ops.Sgate(params[15]) | q[0]
                ops.Sgate(params[16]) | q[1]
                ops.Sgate(params[17]) | q[2]
                ops.Sgate(params[18]) | q[3]
                ops.BSgate(params[19], params[20]) | (q[0], q[1])
                ops.BSgate(params[21], params[22]) | (q[2], q[3])
                ops.BSgate(params[23], params[24]) | (q[1], q[2])
                ops.BSgate(params[25], params[26]) | (q[0], q[1])
                ops.BSgate(params[27], params[28]) | (q[2], q[3])
                ops.BSgate(params[29], params[30]) | (q[1], q[2])
                ops.Rgate(params[31]) | q[0]
                ops.Rgate(params[32]) | q[1]
                ops.Rgate(params[33]) | q[2]
                ops.Dgate(params[34]) | q[0]
                ops.Dgate(params[35]) | q[1]
                ops.Dgate(params[36]) | q[2]
                ops.Dgate(params[37]) | q[3]
                ops.Pgate(params[38]) | q[0]
                ops.Pgate(params[39]) | q[1]
                ops.Pgate(params[40]) | q[2]
                ops.Pgate(params[41]) | q[3]
                ops.Kgate(params[42]) | q[0]
                ops.Kgate(params[43]) | q[1]
                ops.Kgate(params[44]) | q[2]
                ops.Kgate(params[45]) | q[3]

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

            normalization = p0 + p1 + p2 + 1e-10
            output = [p0 / normalization, p1 / normalization, p2 / normalization]

            return output

        circuit_output = [single_input_circuit(x) for x in X]

        return np.array(circuit_output)

    def _myloss(self, circuit_output, targets):

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

    def train(self, lr: learning_rate, sq: squeeze_rate, steps: int, trainX: list, trainY: list) -> None:

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
        self._upload_params()

    def score_model(self, testX: np.array, testY: np.array) -> None:
        test_score = self.learner.score_circuit(X=testX, Y=testY, outputs_to_predictions=self._outputs_to_predictions)
        print("\nPossible scores to print: {}".format(list(test_score.keys())))
        print("Accuracy on test set: {}".format(test_score['accuracy']))
        print("Loss on test set: {}".format(test_score['loss']))

        name = 'FisherIris/results.txt'
        with open(name, 'a') as file:
            file.write('results on ' + str(datetime.datetime.now()) + ' : \n')
            file.write(f'squeezing parameter:    {self.squeeze_param}+\n')
            file.write(f'learning rate:     {self.lr} \n')
            file.write(f'steps:     {self.steps} \n')
            for i in range(len(testY)):
                file.write('x: ' + str(testX[i]) + ', y: ' + str(testY[i]) + '\n')
            file.write("Accuracy on test set: {}".format(test_score['accuracy']) + '\n')
            file.write("Loss on test set: {}".format(test_score['loss']) + '\n\n\n')
