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
        self.params = [make_param(name='param' + str(i), constant=.5) for i in range(54)]
        self.squeeze_rate, self.learner, self.clf_task = None, None, None
        self.lr, self.steps = None, None
        self.step = 0
        self.X, self.Y = None, None
        if params is not None:
            pass

    # @staticmethod
    def _myloss(self, circuit_output, targets):
        # TODO надо свою функцию ошибок сделать.
        # pass
        # return cross_entropy_with_softmax(outputs=circuit_output, targets=targets) / len(targets)
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
            elif len(x) == 16:
                return 4, np.array(x.reshape([4, 4]))
            elif len(x) == 4:
                return 2, np.array(x.reshape([2, 2]))

        def layer_circuit(x, params, delta):
            """
            54 parameters
            :param x: input data with shape (4,)
            :param params: circuit params
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
                ops.BSgate(params[4 + delta], params[5 + delta]) | (q[1], q[2])
                # ops.BSgate(params[6 + delta], params[7 + delta]) | (q[0], q[3])
                # ops.BSgate(params[9 + delta], params[8 + delta]) | (q[2], q[0])
                # ops.BSgate(params[10 + delta], params[11 + delta]) | (q[1], q[3])
                ops.Rgate(params[12 + delta]) | q[0]
                ops.Rgate(params[13 + delta]) | q[1]
                ops.Rgate(params[14 + delta]) | q[2]
                # ops.Sgate(params[15 + delta]) | q[0]
                # ops.Sgate(params[16 + delta]) | q[1]
                # ops.Sgate(params[17 + delta]) | q[2]
                # ops.Sgate(params[18 + delta]) | q[3]
                # # ops.BSgate(params[19 + delta], params[20 + delta]) | (q[0], q[1])
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

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            p0 = state.fock_prob([2, 0, 0, 0])
            p1 = state.fock_prob([0, 2, 0, 0])

            normalization = p0 + p1 + 1e-10  # + p2
            output = p0 / normalization  # , p1 / normalization]  # , p2 / normalization]
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
            # q = [_layer_circuit(x=block, params=params, delta=delta) for block in input_x]
            # return np.array(q)
            return input_x

        def output_layer(x, params, delta):

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
                # ops.BSgate(params[6 + delta], params[7 + delta]) | (q[0], q[3])
                # ops.BSgate(params[9 + delta], params[8 + delta]) | (q[2], q[0])
                # ops.BSgate(params[10 + delta], params[11 + delta]) | (q[1], q[3])
                ops.Rgate(params[12 + delta]) | q[0]
                ops.Rgate(params[13 + delta]) | q[1]
                ops.Rgate(params[14 + delta]) | q[2]
                # ops.Sgate(params[15 + delta]) | q[0]
                # ops.Sgate(params[16 + delta]) | q[1]
                # ops.Sgate(params[17 + delta]) | q[2]
                # ops.Sgate(params[18 + delta]) | q[3]
                # ops.BSgate(params[19 + delta], params[20 + delta]) | (q[0], q[1])
                # ops.BSgate(params[21 + delta], params[22 + delta]) | (q[2], q[3])
                # ops.BSgate(params[23 + delta], params[24 + delta]) | (q[1], q[2])
                # ops.BSgate(params[25 + delta], params[26 + delta]) | (q[0], q[1])
                # ops.BSgate(params[27 + delta], params[28 + delta]) | (q[2], q[3])
                # ops.BSgate(params[29 + delta], params[30 + delta]) | (q[1], q[2])
                # ops.Rgate(params[31 + delta]) | q[0]
                # ops.Rgate(params[32 + delta]) | q[1]
                # ops.Rgate(params[33 + delta]) | q[2]
                ops.Dgate(params[15 + delta]) | q[0]
                ops.Dgate(params[16 + delta]) | q[1]
                ops.Dgate(params[17 + delta]) | q[2]
                ops.Dgate(params[18 + delta]) | q[3]
                ops.Pgate(params[19 + delta]) | q[0]
                ops.Pgate(params[20 + delta]) | q[1]
                ops.Pgate(params[21 + delta]) | q[2]
                ops.Pgate(params[22 + delta]) | q[3]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})
            result = eng.run(qnn)
            state = result.state

            p0 = state.fock_prob([2, 0, 0, 0])
            p1 = state.fock_prob([0, 2, 0, 0])

            normalization = p0 + p1 + 1e-10  # + p2
            output = p0 / normalization  # , p1 / normalization]  # , p2 / normalization]

            return output

        def _single_circuit(x):
            # print(i)
            new_x = layer(x)
            q = [layer_circuit(x=block, params=params, delta=0) for block in new_x]
            new_xx = layer(np.array(q).flatten())
            # new_xx = layer(np.array(x).flatten(), params, delta=54)
            qq = [layer_circuit(x=block, params=params, delta=15) for block in new_xx]
            output = output_layer(np.array(qq).flatten(), params, delta=30)
            # output = output_layer(np.array(x).flatten(), params, delta=108)
            return output

        circuit_output = [_single_circuit(x) for x in X]
        return circuit_output

    def predict(self, data_to_predict) -> np.array:
        outcomes = self.learner.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        predictions = outcomes['predictions']
        return predictions

    def fit(self, lr: float, sq: float, steps: int, clf_task: str, trainX: list, trainY: list) -> None:
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
            self.learner.train_circuit(X=trainX, Y=trainY, steps=steps)
            upload_params(file_name='Mnist/CNN/params.txt', input_name='CNN_params', data=self.params)

    def score_model(self, testX: np.array, testY: np.array) -> None:
        test_score = self.learner.score_circuit(X=testX, Y=testY, outputs_to_predictions=self._outputs_to_predictions)
        print("\nPossible scores to print: {}".format(list(test_score.keys())))
        print("Accuracy on test set: {}".format(test_score['accuracy']))
        print("Loss on test set: {}".format(test_score['loss']))

        name = 'Mnist/CNN/results.txt'
        with open(name, 'a') as file:
            file.write('results on ' + str(datetime.datetime.now()) + ' : \n')
            file.write(f'squeezing parameter:    {self.squeeze_rate}+\n')
            file.write(f'learning rate:     {self.lr} \n')
            file.write(f'steps:     {self.steps} \n')
            for i in range(len(testY)):
                file.write('x: ' + str(testX[i]) + ', y: ' + str(testY[i]) + '\n')
            file.write("Accuracy on test set: {}".format(test_score['accuracy']) + '\n')
            file.write("Loss on test set: {}".format(test_score['loss']) + '\n\n\n')
