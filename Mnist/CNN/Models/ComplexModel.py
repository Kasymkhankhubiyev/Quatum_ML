import strawberryfields as sf
import numpy as np
import datetime

from exceptions import DoesntMatchChosenTask
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss
from helper import upload_params


class Model:
    """
    cf_tasl: classification type, if 'binary' - binary classification,
    'multi' - multiclass classification
    """

    def __init__(self, params=None) -> None:
        self.params = [make_param(name='param' + str(i), constant=.5) for i in range(43)]
        self.squeeze_rate, self.learner, self.clf_task = None, None, None
        self.lr, self.steps = None, None
        self.step = 0
        self.X, self.Y = None, None
        if params is not None:
            pass

    # @staticmethod
    def _myloss(self, circuit_output, targets):
        return square_loss(outputs=circuit_output, targets=targets) / len(targets)

    # @staticmethod
    def _outputs_to_predictions(self, circuit_output):
        return np.round(circuit_output)

    # @staticmethod
    def _circuit(self, X, params):

        def shaper(x) -> tuple[int, np.array]:
            """
            x - a single picture
            :param x: an array of pixels
            :return: shape and reshaped array of pixels
            """
            if len(x) == 64:
                return 8, np.array(x.reshape([8, 8]))
            elif len(x) == 36:
                return 6, np.array(x.reshape([6, 6]))
            elif len(x) == 35:
                return 5, np.array(x.reshape([5, 5]))
            elif len(x) == 16:
                return 4, np.array(x.reshape([4, 4]))
            elif len(x) == 9:
                return 3, np.array(x.reshape([4, 4]))
            elif len(x) == 4:
                return 2, np.array(x.reshape([2, 2]))

        def conv_3x3_layer(x, delta):
            """
            54 parameters
            :param x: input data with shape (4,)
            :param delta: parameters shift for the current layer
            :return: bosons amount in the 0's qumode.
            """
            qnn = sf.Program(9)
            with qnn.context as q:
                ops.Sgate(self.squeeze_rate, x[0]) | q[0]
                ops.Sgate(self.squeeze_rate, x[1]) | q[1]
                ops.Sgate(self.squeeze_rate, x[2]) | q[2]
                ops.Sgate(self.squeeze_rate, x[3]) | q[3]
                ops.Sgate(self.squeeze_rate, x[4]) | q[4]
                ops.Sgate(self.squeeze_rate, x[5]) | q[5]
                ops.Sgate(self.squeeze_rate, x[6]) | q[6]
                ops.Sgate(self.squeeze_rate, x[7]) | q[7]
                ops.Sgate(self.squeeze_rate, x[8]) | q[8]
                ops.BSgate(params[0 + delta], params[1 + delta]) | (q[0], q[1])
                ops.BSgate(params[2 + delta], params[3 + delta]) | (q[2], q[3])
                ops.BSgate(params[4 + delta], params[5 + delta]) | (q[4], q[5])
                ops.BSgate(params[6 + delta], params[7 + delta]) | (q[6], q[7])
                ops.Rgate(params[16 + delta]) | q[7]
                ops.BSgate(params[8 + delta], params[9 + delta]) | (q[7], q[8])
                ops.Rgate(params[17 + delta]) | q[1]
                ops.Rgate(params[18 + delta]) | q[2]
                ops.BSgate(params[10 + delta], params[11 + delta]) | (q[1], q[2])
                ops.Rgate(params[19 + delta]) | q[5]
                ops.Rgate(params[20 + delta]) | q[7]
                ops.BSgate(params[12 + delta], params[13 + delta]) | (q[5], q[7])
                ops.Rgate(params[21 + delta]) | q[5]
                ops.Rgate(params[22 + delta]) | q[2]
                ops.BSgate(params[14 + delta], params[15 + delta]) | (q[2], q[5])
                # ops.Rgate(params[16 + delta]) | q[0]
                # ops.Rgate(params[17 + delta]) | q[2]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            modes = 9*[0]
            modes[2] = 2
            p0 = state.fock_prob(modes)
            modes[2] = 0
            modes[5] = 2
            p1 = state.fock_prob(modes)

            normalization = p0 + p1 + 1e-10  # + p2
            output = p0 / normalization  # , p1 / normalization]  # , p2 / normalization]
            return output

        def conv_2x2_layer(x, delta):
            """
            54 parameters
            :param x: input data with shape (4,)
            :param delta: parameters shift for the current layer
            :return: bosons amount in the 0's qumode.
            """
            qnn = sf.Program(4)
            with qnn.context as q:
                ops.Sgate(self.squeeze_rate, x[0]) | q[0]
                ops.Sgate(self.squeeze_rate, x[1]) | q[1]
                ops.Sgate(self.squeeze_rate, x[2]) | q[2]
                ops.Sgate(self.squeeze_rate, x[3]) | q[3]
                ops.BSgate(params[0 + delta], params[1 + delta]) | (q[0], q[1])
                ops.BSgate(params[2 + delta], params[3 + delta]) | (q[2], q[3])
                ops.Rgate(params[4 + delta]) | q[0]
                ops.Rgate(params[5 + delta]) | q[2]
                ops.BSgate(params[6 + delta], params[7 + delta]) | (q[0], q[2])
                # ops.Rgate(params[8 + delta]) | q[0]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            p0 = state.fock_prob([2, 0, 0, 0])
            p1 = state.fock_prob([0, 2, 0, 0])

            normalization = p0 + p1 + 1e-10  # + p2
            output = p0 / normalization  # , p1 / normalization]  # , p2 / normalization]
            return output

        def make_matrixes_3x3(x):
            """
                идем матрицей 3х3 с шагом 1,
                тогда из мтарицы 8х8 получаем 6х6
            """
            axs_scale, _x = shaper(x)
            input_x = []
            for i in range(0, axs_scale-2, 1):  # x
                for j in range(0, axs_scale-2, 1):  # y
                    input_x.append(np.array([_x[i, j], _x[i, j + 1], _x[i, j + 2],
                                             _x[i + 1, j], _x[i + 1, j + 1], _x[i + 1, j + 2],
                                             _x[i + 2, j], _x[i + 2, j + 1], _x[i + 2, j + 2],
                                             ]))

            input_x = np.array(input_x)
            return input_x

        def make_matrixes_2x2(x):
            """
            картинка 8x8 -> маска 2х2 с шагом 1
            :param x:
            :return:
            """
            axs_scale, _x = shaper(x)
            input_x = []
            for i in range(0, axs_scale-1, 1):  # x
                for j in range(0, axs_scale-1, 1):  # y
                    input_x.append(np.array([_x[i, j], _x[i, j + 1], _x[i + 1, j], _x[i + 1, j + 1]]))

            input_x = np.array(input_x)
            return input_x

        def max_pooling_2x2(x):
            _x = make_matrixes_2x2(x)
            return np.max(_x, axis=1)

        def full_con_layer(x, delta):

            qnn = sf.Program(4)
            # print('output layer')

            with qnn.context as q:
                ops.Sgate(self.squeeze_rate, x[0]) | q[0]
                ops.Sgate(self.squeeze_rate, x[1]) | q[1]
                ops.Sgate(self.squeeze_rate, x[2]) | q[2]
                ops.Sgate(self.squeeze_rate, x[3]) | q[3]
                ops.BSgate(params[0 + delta], params[1 + delta]) | (q[0], q[1])
                ops.BSgate(params[2 + delta], params[3 + delta]) | (q[2], q[3])
                ops.BSgate(params[4 + delta], params[5 + delta]) | (q[1], q[2])
                # ops.Sgate(params[6 + delta]) | q[0]
                # ops.Sgate(params[7 + delta]) | q[1]
                # ops.Sgate(params[8 + delta]) | q[2]
                # ops.Sgate(params[9 + delta]) | q[3]
                # ops.MZgate(params[10 + delta], params[11 + delta]) | (q[0], q[1])
                # ops.MZgate(params[12 + delta], params[13 + delta]) | (q[2], q[3])
                # ops.MZgate(params[14 + delta], params[15 + delta]) | (q[1], q[2])
                ops.Rgate(params[6 + delta]) | q[1]
                ops.Rgate(params[7 + delta]) | q[2]
                # ops.Dgate(params[16 + delta]) | q[0]
                ops.Dgate(params[8 + delta]) | q[1]
                ops.Dgate(params[9 + delta]) | q[2]
                # ops.Dgate(params[19 + delta]) | q[3]
                # ops.Pgate(params[20 + delta]) | q[0]
                ops.Pgate(params[10 + delta]) | q[1]
                ops.Pgate(params[11 + delta]) | q[2]
                # ops.Pgate(params[23 + delta]) | q[3]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            p0 = state.fock_prob([0, 2, 0, 0])
            p1 = state.fock_prob([0, 0, 2, 0])

            normalization = p0 + p1 + 1e-10  # + p2
            output = p0 / normalization  # , p1 / normalization]  # , p2 / normalization]

            return output

        def _single_circuit(x):
            _x = make_matrixes_3x3(x)
            _x = [conv_3x3_layer(x=block, delta=0) for block in _x]
            _x = max_pooling_2x2(np.array(_x).flatten())
            _x = make_matrixes_3x3(np.array(_x).flatten())
            _x = [conv_2x2_layer(x=block, delta=23) for block in _x]
            output = full_con_layer(np.array(_x).flatten(), delta=31)
            return output

        circuit_output = [_single_circuit(x) for x in X]
        return circuit_output

    def predict(self, data_to_predict) -> np.array:
        outcomes = self.learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions

    def fit(self, lr: float, sq: float, steps: int, clf_task: str, train_x: list, train_y: list) -> None:
        """
        В задаче сверточных сетей мы обучаем сверточную матрицу:
        :return:
        """
        self.lr, self.steps, self.squeeze_rate = lr, steps, sq
        if clf_task not in ['binary', 'multi']:
            raise DoesntMatchChosenTask(tasks_list=['binary', 'multi'], err_task=clf_task)
        else:
            self.clf_task = clf_task
        if self.clf_task == 'binary':
            hyperparams = {'circuit': self._circuit,
                           'init_circuit_params': self.params,
                           'task': 'supervised',
                           'loss': self._myloss,
                           'optimizer': 'SGD',
                           'init_learning_rate': lr,
                           'log_every': 1,
                           'warm_start': False}

            self.learner = CircuitLearner(hyperparams=hyperparams)
            self.learner.train_circuit(X=train_x, Y=train_y, steps=steps)
            upload_params(file_name='Mnist/CNN/params.txt', input_name='CNN_params', data=self.params)

    def score_model(self, test_x: np.array, test_y: np.array) -> None:
        test_score = self.learner.score_circuit(X=test_x, Y=test_y, outputs_to_predictions=self._outputs_to_predictions)
        print("\nPossible scores to print: {}".format(list(test_score.keys())))
        print("Accuracy on test set: {}".format(test_score['accuracy']))
        print("Loss on test set: {}".format(test_score['loss']))

        name = 'Mnist/CNN/results.txt'
        with open(name, 'a') as file:
            file.write('results on ' + str(datetime.datetime.now()) + ' : \n')
            file.write(f'different convolutions:    {2}\n')
            file.write(f'squeezing parameter:    {self.squeeze_rate}+\n')
            file.write(f'learning rate:     {self.lr} \n')
            file.write(f'steps:     {self.steps} \n')
            file.write("Accuracy on test set: {}".format(test_score['accuracy']) + '\n')
            file.write("Loss on test set: {}".format(test_score['loss']) + '\n\n\n')
