import strawberryfields as sf
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss

sqeeze_param = 0.195

class Model:

    def __init__(self, params):
        self.params = [make_param(name='phi'+str(i), constant=.7) for i in range(9)]

    def _circuit(self, X, params):

        # prog = sf.Program(2)
        sq = sqeeze_param

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

    def train(self, lr: float, steps: int, trainX, trainY) -> None:
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

    def predict(self, data):
        outcomes = self.learner.run_circuit(X=data, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions

