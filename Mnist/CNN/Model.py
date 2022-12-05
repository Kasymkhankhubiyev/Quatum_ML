import strawberryfields as sf
import numpy as np

from exceptions import DoesntMatchChosenTask
from strawberryfields import ops
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param


class Model:
    """
    cf_tasl: classification type, if 'binary' - binary classification,
    'multi' - multiclass classification
    """

    def __init__(self, clf_task: str, params=None,) -> None:
        self.params_layer0, self.squeeze_rate = None, None
        self.learner, self.clf_task = None, clf_task
        self.params_layer1, self.params_layer2 = None, None
        if clf_task not in ['binary', 'multi']:
            raise DoesntMatchChosenTask(tasks_list=['binary', 'multi'], err_task=clf_task)

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
            # ops.Pgate(params[41]) | q[3]
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

    def _layer(self, x, params):
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

    def _output_layer(self, x, params):
        pass

    # надо возвращать сам регистр
    def _circuit(self, x):
        sq = self.squeeze_rate

        def _single_circuit(x):

            q = self._layer(x, self.params_layer0)
            q = self._layer(q, self.params_layer1)
            q = self._layer(q, self.params_layer1)


        pass

    def predict(self) -> np.array:
        pass

    def fit(self) -> None:
        if self.clf_task == 'binary':
            pass
