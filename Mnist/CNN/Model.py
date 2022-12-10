import strawberryfields as sf
import numpy as np
import datetime

from exceptions import DoesntMatchChosenTask
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import cross_entropy_with_softmax


class Model:
    """
    cf_tasl: classification type, if 'binary' - binary classification,
    'multi' - multiclass classification
    """

    def __init__(self, params=None,) -> None:
        self.params, self.squeeze_rate = [make_param(name='param' + str(i), constant=.5) for i in range(52)], None
        self.learner, self.clf_task = None, clf_task
        self.lr, self.steps = None, None
        if params is not None:
            pass

    def _shaper(self, x) -> tuple[int, np.array]:
        """
        x - a single picture
        :param x:
        :return:
        """
        if len(x) == 64:
            return 8, np.array(x.reshape([8, 8]))
        elif len(x) == 16:
            return 4, np.array(x.reshape([4, 4]))
        elif len(x) == 4:
            return 2, np.array(x.reshape([2, 2]))

    def _layer_circuit(self, x, params):
        sq = self.squeeze_rate
        qnn = sf.Program(4)

        with qnn.context as q:
            ops.Sgate(sq, x[0]) | q[0]
            ops.Sgate(sq, x[1]) | q[1]
            ops.Sgate(sq, x[2]) | q[2]
            ops.Sgate(sq, x[3]) | q[3]
            ops.BSgate(params[0], params[1]) | (q[0], q[1])
            ops.BSgate(params[2], params[3]) | (q[2], q[3])
            ops.BSgate(params[4], params[5]) | (q[1], q[2])
            ops.BSgate(params[6], params[7]) | (q[0], q[3])
            ops.BSgate(params[9], params[8]) | (q[2], q[0])
            ops.BSgate(params[10], params[11]) | (q[1], q[3])
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
            # ops.Kgate(params[42]) | q[0]
            # ops.Kgate(params[43]) | q[1]
            # ops.Kgate(params[44]) | q[2]
            # ops.Kgate(params[45]) | q[3]

            ops.BSgate(params[41], params[42]) | (q[0], q[1])
            ops.BSgate(params[44], params[43]) | (q[2], q[3])
            ops.BSgate(params[45], params[46]) | (q[1], q[2])
            ops.BSgate(params[47], params[48]) | (q[0], q[1])
            ops.BSgate(params[49], params[50]) | (q[2], q[3])
            ops.BSgate(params[51], params[51]) | (q[1], q[2])

        return q[2]

    def _layer(self, x, params, disp):
        """
        8X8 - 64 // 4 = 16 блоков
        :param x: a single picture
        :param q:
        :return:
        """
        axs_scale, _x = self._shaper(x)
        input_x = []
        for i in range(0, axs_scale//2, 2):  # x
            for j in range(0, axs_scale//2, 2):  # y
                input_x.append(np.array[_x[i, j], _x[i, j+1], _x[i+1, j], _x[i+1, j+1]])

        input_x = np.array(input_x)
        q = []

        for block in input_x:
            q.append(self._layer_circuit(x=block, params=params))

        return q

    def _output_layer(self, x, params, disp):

        sq = self.squeeze_rate
        qnn = sf.Program(4)

        with qnn.context as q:
            ops.Sgate(sq, x[0]) | q[0]
            ops.Sgate(sq, x[1]) | q[1]
            ops.Sgate(sq, x[2]) | q[2]
            ops.Sgate(sq, x[3]) | q[3]
            ops.BSgate(params[0], params[1]) | (q[0], q[1])
            ops.BSgate(params[2], params[3]) | (q[2], q[3])
            ops.BSgate(params[4], params[5]) | (q[1], q[2])
            ops.BSgate(params[6], params[7]) | (q[0], q[3])
            ops.BSgate(params[9], params[8]) | (q[2], q[0])
            ops.BSgate(params[10], params[11]) | (q[1], q[3])
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

        eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})

        result = eng.run(qnn)
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

    # надо возвращать сам регистр
    def _circuit(self, X):
        sq = self.squeeze_rate

        def _single_circuit(x):

            q = self._layer(x, self.params)
            print(f'layer 0: {q}')
            q = self._layer(q, self.params)
            print(f'layer 1: {q}')
            # q = self._layer(q, self.params_layer2)
            # print(f'layer 0: {q}')

            return self._output_layer(q, self.params)

        predictions = [_single_circuit(x) for x in X]

        return np.array(predictions)

    def _myloss(self, circuit_output, targets):

        return cross_entropy_with_softmax(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        for i in range(len(circuit_output)):
            circuit_output[i] = round(circuit_output[i])
        return circuit_output

    def predict(self) -> np.array:
        pass

    def _upload_params(self):
        name = 'FisherIris/params_on_'+(datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_M"))+'.txt'
        with open(name, 'a') as file:
            file.write('Mnist/CNN/params_on_'+datetime.datetime.strftime(datetime.datetime.now())+'\n\n')
            for i in range(len(self.params)):
                file.write(str(self.params[i])+',')
            file.write('\n\n\n')

    def fit(self, lr: float, sq: float, steps: int, clf_task: str, trainX: list, trainY: list) -> None:
        """
        В задаче сверточных сетей мы обучаем сверточную матрицу:
        :return:
        """
        self.lr = lr
        self.steps = steps
        if clf_task not in ['binary', 'multi']:
            raise DoesntMatchChosenTask(tasks_list=['binary', 'multi'], err_task=clf_task)
        if self.clf_task == 'binary':
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
            file.write(f'squeezing parameter:    {self.squeeze_rate}+\n')
            file.write(f'learning rate:     {self.lr} \n')
            file.write(f'steps:     {self.steps} \n')
            for i in range(len(testY)):
                file.write('x: ' + str(testX[i]) + ', y: ' + str(testY[i]) + '\n')
            file.write("Accuracy on test set: {}".format(test_score['accuracy']) + '\n')
            file.write("Loss on test set: {}".format(test_score['loss']) + '\n\n\n')

