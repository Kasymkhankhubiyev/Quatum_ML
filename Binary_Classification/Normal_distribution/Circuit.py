import strawberryfields as sf
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss
import datetime
from Binary_Classification.Normal_distribution.VisualizeData import *

squeeze = float


class Model:

    def __init__(self):
        self.params = [make_param(name='phi'+str(i), constant=.7) for i in range(9)]
        self.learner, self.squeeze_param, self.lr, self.steps = None, None, None, None
        self.transform_data = []
        self.counter = 0

    def _circuit(self, X, params):
        # print(self.counter)
        # self.counter += 1
        # prog = sf.Program(2)
        sq = self.squeeze_param

        # transform_data = []

        def single_input_circuit(x):
            prog = sf.Program(2)

            with prog.context as q:
                # ops.Dgate(x[0], 0.) | q[0]
                # ops.Dgate(x[1], 0.) | q[1]
                ops.Sgate(sq, x[0]) | q[0]
                ops.Sgate(sq, x[1]) | q[1]
                ops.BSgate(params[0], params[7]) | (q[0], q[1])
                ops.Dgate(params[1]) | q[0]
                ops.Dgate(params[2]) | q[1]
                ops.Pgate(params[3]) | q[0]
                ops.Pgate(params[4]) | q[1]
                ops.Kgate(params[5]) | q[0]
                ops.Kgate(params[6]) | q[1]

                # ops.MeasureFock() | q[0]
                # ops.MeasureFock() | q[1]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})

            result = eng.run(prog)

            state = result.state

            p0 = state.fock_prob([0, 2])
            p1 = state.fock_prob([2, 0])
            normalization = p0 + p1 + 1e-10
            output = p1 / normalization

            return output

        circuit_output = [single_input_circuit(x) for x in X]

        return circuit_output

    def _myloss(self, circuit_output, targets):
        return square_loss(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        return round(circuit_output)

    def _upload_params(self):
        name = 'Binary_Classification/Normal_distribution/params_on_'+ \
            (datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d"))+'.txt'
        with open(name, 'a') as file:
            for i in range(len(self.params)):
                file.write(str(self.params[i])+',')
                file.write('\n\n\n')

    def train(self, lr: float, steps: int, sq: squeeze, trainX, trainY) -> None:
        self.squeeze_param = sq
        self.lr, self.steps = lr, steps
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': lr,

                       'log_every': 1,
                       'warm_start': False
                       }  # 'decay': 0.01,

        self.learner = CircuitLearner(hyperparams=hyperparams)
        self.learner.train_circuit(X=trainX, Y=trainY, steps=steps)
        self._upload_params()

    def predict(self, data_to_predict):
        outcomes = self.learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions

    def score_model(self, testX, testY):
        test_score = self.learner.score_circuit(X=testX, Y=testY, outputs_to_predictions=self._outputs_to_predictions)
        print("\nPossible scores to print: {}".format(list(test_score.keys())))
        print("Accuracy on test set: {}".format(test_score['accuracy']))
        print("Loss on test set: {}".format(test_score['loss']))

        name = 'Binary_Classification/Normal_distribution/results.txt'
        with open(name, 'a') as file:
            file.write('results on '+str(datetime.datetime.now()) + ' : \n')
            file.write(f'squeezing parameter:    {self.squeeze_param}+\n')
            file.write(f'learning rate:     {self.lr} \n')
            file.write(f'steps:     {self.steps} \n')
            for i in range(len(testY)):
                file.write('x: '+str(testX[i]) + ', y: '+str(testY[i]) + '\n')
            file.write("Accuracy on test set: {}".format(test_score['accuracy']) + '\n')
            file.write("Loss on test set: {}".format(test_score['loss']) + '\n\n\n')

    def _project_data(self, data):
        sq = self.squeeze_param

        transform_data = []

        for x in data:
            prog = sf.Program(2)

            with prog.context as q:
                # ops.Dgate(x[0], 0.) | q[0]
                # ops.Dgate(x[1], 0.) | q[1]
                ops.Sgate(sq, x[0]) | q[0]
                ops.Sgate(sq, x[1]) | q[1]
                ops.BSgate(self.params[0]['val'], self.params[7]['val']) | (q[0], q[1])
                ops.Dgate(self.params[1]['val']) | q[0]
                ops.Dgate(self.params[2]['val']) | q[1]
                ops.Pgate(self.params[3]['val']) | q[0]
                ops.Pgate(self.params[4]['val']) | q[1]
                ops.Kgate(self.params[5]['val']) | q[0]
                ops.Kgate(self.params[6]['val']) | q[1]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            eng.run(prog)

        return np.array(transform_data)
