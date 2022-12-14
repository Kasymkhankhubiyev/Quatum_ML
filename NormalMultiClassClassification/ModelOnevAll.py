import strawberryfields as sf
from strawberryfields import ops
import numpy as np
from qmlt.numerical import CircuitLearner
from qmlt.numerical.helpers import make_param
from qmlt.numerical.losses import square_loss
from qmlt.numerical.losses import cross_entropy_with_softmax
import datetime
from NormalMultiClassClassification.DataPrep import DataSet

squeeze_rate = float
learning_rate = float


class Model:
    def __init__(self) -> None:
        self.lr, self.steps, self.squeeze_param = None, None, None
        self.learner0, self.learner1, self.learner2, self.learner3 = None, None, None, None
        self.params0 = [make_param(name='param' + str(i), constant=.7) for i in range(9)]
        self.params1 = [make_param(name='param' + str(i), constant=.7) for i in range(9)]
        self.params2 = [make_param(name='param' + str(i), constant=.7) for i in range(9)]
        self.params3 = [make_param(name='param' + str(i), constant=.7) for i in range(9)]

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
                # ops.Kgate(params[5]) | q[0]
                # ops.Kgate(params[6]) | q[1]

            eng = sf.Engine('fock', backend_options={'cutoff_dim': 5, 'eval': True})

            result = eng.run(prog)

            state = result.state

            p0 = state.fock_prob([0, 2])
            p1 = state.fock_prob([2, 0])
            normalization = p0 + p1 + 1e-10
            output = (p0 / normalization)  # * np.sqrt(x[0]**2 + x[1]**2)

            return output

        circuit_output = np.array([single_input_circuit(x) for x in X])
        # summ = np.sum(np.array([np.sqrt(x[0]**2 + x[1]**2) for x in X]))

        return circuit_output

    def _myloss(self, circuit_output, targets):
        return square_loss(outputs=circuit_output, targets=targets) / len(targets)
        # return cross_entropy_with_softmax(outputs=circuit_output, targets=targets) / len(targets)

    def _outputs_to_predictions(self, circuit_output):
        return (circuit_output)
        # return round(circuit_output)

    def _upload_params(self):
        name = 'NormalMultiClassClassification/params.txt'
        with open(name, 'a') as file:
            for i in range(len(self.params0)):
                file.write(str(self.params0[i])+',')
            file.write('\n\n')
            for i in range(len(self.params1)):
                file.write(str(self.params1[i])+',')
            file.write('\n\n')
            for i in range(len(self.params2)):
                file.write(str(self.params2[i])+',')
            file.write('\n\n')
            for i in range(len(self.params3)):
                file.write(str(self.params3[i])+',')
            file.write('\n\n\n')

    def _train_first_vs_all(self, lr, trainX, trainY, steps):
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params0,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': lr,
                       'log_every': 1,
                       'warm_start': False
                       }
        self.learner0 = CircuitLearner(hyperparams=hyperparams)
        self.learner0.train_circuit(X=trainX, Y=trainY, steps=steps)

    def _train_second_vs_all(self, lr, trainX, trainY, steps):
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params1,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': lr,
                       'log_every': 1,
                       'warm_start': False
                       }
        self.learner1 = CircuitLearner(hyperparams=hyperparams)
        self.learner1.train_circuit(X=trainX, Y=trainY, steps=steps)

    def _train_third_vs_all(self, lr, trainX, trainY, steps):
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params2,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': lr,
                       'log_every': 1,
                       'warm_start': False
                       }
        self.learner2 = CircuitLearner(hyperparams=hyperparams)
        self.learner2.train_circuit(X=trainX, Y=trainY, steps=steps)

    def _train_fourth_vs_all(self, lr, trainX, trainY, steps):
        hyperparams = {'circuit': self._circuit,
                       'init_circuit_params': self.params3,
                       'task': 'supervised',
                       'loss': self._myloss,
                       'optimizer': 'SGD',
                       'init_learning_rate': lr,
                       'log_every': 1,
                       'warm_start': False
                       }
        self.learner3 = CircuitLearner(hyperparams=hyperparams)
        self.learner3.train_circuit(X=trainX, Y=trainY, steps=steps)

        self._upload_params()

    def train(self, lr: learning_rate, steps: int, sq: squeeze_rate, dataset: DataSet) -> None:
        self.squeeze_param = sq
        self.lr, self.steps = lr, steps

        # 0 class vs all
        train_x = np.vstack((dataset.trainX_0, dataset.trainX_1, dataset.trainX_2, dataset.trainX_3))
        train_y = np.hstack([[0] * (len(dataset.trainX_0)),
                             [1] * (len(dataset.trainX_1) + len(dataset.trainX_2) + len(dataset.trainX_3))])
        self._train_first_vs_all(lr=lr, trainX=train_x, trainY=train_y, steps=steps)

        # 1 class vs all
        train_x = np.vstack((dataset.trainX_1, dataset.trainX_0, dataset.trainX_2, dataset.trainX_3))
        train_y = np.hstack([[0] * (len(dataset.trainX_1)),
                             [1] * (len(dataset.trainX_0) + len(dataset.trainX_2) + len(dataset.trainX_3))])
        self._train_second_vs_all(lr=lr, trainX=train_x, trainY=train_y, steps=steps)

        # 2 class vs all
        train_x = np.vstack((dataset.trainX_2, dataset.trainX_0, dataset.trainX_1, dataset.trainX_3))
        train_y = np.hstack([[0] * (len(dataset.trainX_2)),
                             [1] * (len(dataset.trainX_0) + len(dataset.trainX_1) + len(dataset.trainX_3))])
        self._train_third_vs_all(lr=lr, trainX=train_x, trainY=train_y, steps=steps)

        # 3 class vs all
        train_x = np.vstack((dataset.trainX_3, dataset.trainX_0, dataset.trainX_1, dataset.trainX_2))
        train_y = np.hstack([[0] * (len(dataset.trainX_3)),
                             [1] * (len(dataset.trainX_0) + len(dataset.trainX_1) + len(dataset.trainX_2))])
        self._train_fourth_vs_all(lr=lr, trainX=train_x, trainY=train_y, steps=steps)

    def _predict_class(self, data_to_predict):
        outcomes = self.learner0.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        prediction0 = outcomes['predictions']

        outcomes = self.learner1.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        prediction1 = outcomes['predictions']

        outcomes = self.learner2.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        prediction2 = outcomes['predictions']

        outcomes = self.learner3.run_circuit(X=data_to_predict, outputs_to_predictions=self._outputs_to_predictions)
        prediction3 = outcomes['predictions']

        predictions = np.vstack((np.array(prediction0), np.array(prediction1),
                                np.array(prediction2), np.array(prediction3)))
        # print(predictions.T)
        predict = [np.argmax(prediction) for prediction in predictions.T]
        print(predict)
        return predict  # predictions

    def predict(self, data_to_predict):
        predictions = self._predict_class(data_to_predict)
        return predictions

    def score_model(self, testX, testY):
        predictions = self._predict_class(testX)
        counter = 0
        for i in range(len(testY)):
            if predictions[i] == testY[i]:
                counter += 1
        accuracy = counter/len(testY)
        return accuracy

    def score_model_simmetric(self, testX, testY):
        predictions = self._predict_class(testX)
        counter = 0
        for i in range(len(testY)):
            if testY[i] == 0 or testY[i] == 1:
                if predictions[i] == 0 or predictions[i] == 1:
                    counter += 1
            if testY[i] == 2 or testY[i] == 3:
                if predictions[i] == 2 or predictions[i] == 3:
                    counter += 1
        accuracy = counter / len(testY)
        return accuracy
