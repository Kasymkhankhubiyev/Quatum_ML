import strawberryfields as sf
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss
import datetime
from helper import upload_params

squeeze = float


class Model:

    def __init__(self):
        self.params = [make_param(name='phi'+str(i), constant=.7) for i in range(8)]
        self.learner, self.squeeze_param, self.lr, self.steps = None, None, None, None
        self.counter = 0

    def _circuit(self, X, params):
        sq = self.squeeze_param

        def single_input_circuit(x):
            prog = sf.Program(2)

            with prog.context as q:
                ops.Sgate(sq, x[0]) | q[0]
                ops.Sgate(sq, x[1]) | q[1]
                ops.BSgate(params[0], params[7]) | (q[0], q[1])
                ops.Dgate(params[1]) | q[0]
                ops.Dgate(params[2]) | q[1]
                ops.Pgate(params[3]) | q[0]
                ops.Pgate(params[4]) | q[1]
                ops.Kgate(params[5]) | q[0]
                ops.Kgate(params[6]) | q[1]

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

    def _myloss(self, circuit_output, targets):  # в лосс функции навверняка нужно округлить
        return square_loss(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        return round(circuit_output)

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
        upload_params(file_name='Binary_Classification/Normal_distribution/params.txt',
                      input_name='BinarClf_params', data=self.params)

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
