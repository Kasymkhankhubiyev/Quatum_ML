import strawberryfields as sf
import numpy as np

from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss
from helper import upload_params


class CNN:
    def __init__(self):
        self.params = [make_param(name='param' + str(i), constant=.5) for i in range(150)]
        self.squeeze_rate, self.lr, self.steps, self.learner = None, None, None, None
        self.counter = 0

    def _myloss(self, circuit_output, targets):  # в лосс функции навверняка нужно округлить
        return square_loss(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        return round(circuit_output)

    def circuit(self, X, params):

        def shaper(x) -> tuple[int, np.array]:
            """
            x - a single picture
            :param x: an array of pixels
            :return: shape and reshaped array of pixels
            """
            print(self.counter)
            self.counter += 1
            if len(x) == 64:
                return 8, np.array(x.reshape([8, 8]))
            elif len(x) == 16:
                return 4, np.array(x.reshape([4, 4]))
            elif len(x) == 4:
                return 2, np.array(x.reshape([2, 2]))

        def layer_circuit(x, params):
            """
            54 parameters
            :param x: input data with shape (4,)
            :param params: circuit params
            :param delta: parameters shift for the current layer
            :return: bosons amount in the 0's qumode.
            """
            delta = 54
            qnn = sf.Program(4)
            with qnn.context as q:
                ops.Sgate(self.squeeze_rate, x[0]) | q[0]
                ops.Sgate(self.squeeze_rate, x[1]) | q[1]
                ops.Sgate(self.squeeze_rate, x[2]) | q[2]
                ops.Sgate(self.squeeze_rate, x[3]) | q[3]
                ops.BSgate(params[0 + delta], params[1 + delta]) | (q[0], q[1])
                ops.BSgate(params[2 + delta], params[3 + delta]) | (q[2], q[3])
                ops.BSgate(params[4 + delta], params[5 + delta]) | (q[1], q[2])
                ops.BSgate(params[6 + delta], params[7 + delta]) | (q[0], q[3])
                ops.BSgate(params[9 + delta], params[8 + delta]) | (q[2], q[0])
                ops.BSgate(params[10 + delta], params[11 + delta]) | (q[1], q[3])
                ops.Rgate(params[12 + delta]) | q[0]
                ops.Rgate(params[13 + delta]) | q[1]
                ops.Rgate(params[14 + delta]) | q[2]
                ops.Sgate(params[15 + delta]) | q[0]
                ops.Sgate(params[16 + delta]) | q[1]
                ops.Sgate(params[17 + delta]) | q[2]
                ops.Sgate(params[18 + delta]) | q[3]
                # ops.BSgate(params[19 + delta], params[20 + delta]) | (q[0], q[1])
                # ops.BSgate(params[21 + delta], params[22 + delta]) | (q[2], q[3])
                # ops.BSgate(params[23 + delta], params[24 + delta]) | (q[1], q[2])
                # ops.BSgate(params[25 + delta], params[26 + delta]) | (q[0], q[1])
                # ops.BSgate(params[27 + delta], params[28 + delta]) | (q[2], q[3])
                # ops.BSgate(params[29 + delta], params[30 + delta]) | (q[1], q[2])
                # ops.Rgate(params[31 + delta]) | q[0]
                # ops.Rgate(params[32 + delta]) | q[1]
                # ops.Rgate(params[33 + delta]) | q[2]
                # ops.Dgate(params[34 + delta]) | q[0]
                # ops.Dgate(params[35 + delta]) | q[1]
                # ops.Dgate(params[36 + delta]) | q[2]
                # ops.Dgate(params[37 + delta]) | q[3]
                # ops.Pgate(params[38 + delta]) | q[0]
                # ops.Pgate(params[39 + delta]) | q[1]
                # ops.Pgate(params[40 + delta]) | q[2]
                # ops.Pgate(params[41 + delta]) | q[3]
                # ops.Kgate(params[42]) | q[0]
                # ops.Kgate(params[43]) | q[1]
                # ops.Kgate(params[44]) | q[2]
                # ops.Kgate(params[45]) | q[3]
                # ops.BSgate(params[42 + delta], params[43 + delta]) | (q[0], q[1])
                # ops.BSgate(params[44 + delta], params[45 + delta]) | (q[2], q[3])
                # ops.BSgate(params[46 + delta], params[47 + delta]) | (q[1], q[2])
                # ops.BSgate(params[48 + delta], params[49 + delta]) | (q[0], q[1])
                # ops.BSgate(params[50 + delta], params[51 + delta]) | (q[2], q[3])
                # ops.BSgate(params[52 + delta], params[53 + delta]) | (q[1], q[2])
                # ops.MeasureFock() | q[0]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 10, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            p0 = state.fock_prob([0, 2, 0, 0])
            p1 = state.fock_prob([2, 0, 0, 0])
            normalization = p0 + p1 + 1e-10
            output = p1 / normalization
            return output

        def layer(x):
            """
            8X8 - 64 // 4 = 16 блоков
            :param x: a single picture
            :param q:
            :return:
            """
            axs_scale, _x = shaper(x)
            input_x = []
            for i in range(0, axs_scale, 2):  # x
                for j in range(0, axs_scale, 2):  # y
                    input_x.append(np.array([_x[i, j], _x[i, j + 1], _x[i + 1, j], _x[i + 1, j + 1]]))

            input_x = np.array(input_x)
            return input_x

        def output_layer(x):

            # print('output layer')
            # print(self.counter)
            # self.counter += 1
            qnn = sf.Program(4)
            delta = 108

            with qnn.context as q:
                ops.Sgate(self.squeeze_rate, x[0]) | q[0]
                ops.Sgate(self.squeeze_rate, x[1]) | q[1]
                ops.Sgate(self.squeeze_rate, x[2]) | q[2]
                ops.Sgate(self.squeeze_rate, x[3]) | q[3]
                ops.BSgate(params[0 + delta], params[1 + delta]) | (q[0], q[1])
                ops.BSgate(params[2 + delta], params[3 + delta]) | (q[2], q[3])
                ops.BSgate(params[4 + delta], params[5 + delta]) | (q[1], q[2])
                # ops.BSgate(params[6 + delta], params[7 + delta]) | (q[0], q[3])
                # ops.BSgate(params[9 + delta], params[8 + delta]) | (q[2], q[0])
                # ops.BSgate(params[10 + delta], params[11 + delta]) | (q[1], q[3])
                ops.Rgate(params[12 + delta]) | q[0]
                ops.Rgate(params[13 + delta]) | q[1]
                ops.Rgate(params[14 + delta]) | q[2]
                ops.Rgate(params[15 + delta]) | q[3]
                # ops.Sgate(params[15 + delta]) | q[0]
                # ops.Sgate(params[16 + delta]) | q[1]
                # ops.Sgate(params[17 + delta]) | q[2]
                # ops.Sgate(params[18 + delta]) | q[3]
                ops.BSgate(params[19 + delta], params[20 + delta]) | (q[0], q[1])
                ops.BSgate(params[21 + delta], params[22 + delta]) | (q[2], q[3])
                ops.BSgate(params[23 + delta], params[24 + delta]) | (q[1], q[2])
                # ops.BSgate(params[25 + delta], params[26 + delta]) | (q[0], q[1])
                # ops.BSgate(params[27 + delta], params[28 + delta]) | (q[2], q[3])
                # ops.BSgate(params[29 + delta], params[30 + delta]) | (q[1], q[2])
                ops.Rgate(params[31 + delta]) | q[0]
                ops.Rgate(params[32 + delta]) | q[1]
                ops.Rgate(params[33 + delta]) | q[2]
                ops.Dgate(params[34 + delta]) | q[0]
                ops.Dgate(params[35 + delta]) | q[1]
                ops.Dgate(params[36 + delta]) | q[2]
                ops.Dgate(params[37 + delta]) | q[3]
                ops.Pgate(params[38 + delta]) | q[0]
                ops.Pgate(params[39 + delta]) | q[1]
                ops.Pgate(params[40 + delta]) | q[2]
                ops.Pgate(params[41 + delta]) | q[3]

                eng = sf.Engine('fock', backend_options={'cutoff_dim': 10, 'eval': True})
                result = eng.run(qnn)
                state = result.state

                p0 = state.fock_prob([0, 2, 0, 0])
                p1 = state.fock_prob([2, 0, 0, 0])
                normalization = p0 + p1 + 1e-10
                output = p1 / normalization

                return output

        def single_input_circuit(x):
            new_x = layer(x)
            q = [layer_circuit(x, params) for x in new_x]
            return output_layer(np.array(q).flatten())
            # return output_layer(x)

        circuit_output = [single_input_circuit(x) for x in X]
        # circuit_output = [output_layer(x) for x in X]
        return circuit_output

    def fit(self, trainX, trainY, lr, steps, sq):
        self.lr = lr
        self.steps = steps
        self.squeeze_rate = sq

        hyperparams = {'circuit': self.circuit,
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
