import unittest
import numpy as np

from NNClassifier import Layer
from NNClassifier import NeuralNetworkClassifier
from functions import sigmoid_activation, tanh_activation


class LayerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 4, 5]]).transpose()
        t = np.array([[.1, .05]]).transpose()
        alpha = 0.01
        sigmoid, sigmoid_derivative = sigmoid_activation
        theta2 = np.array([[.1, .3, .5],
                           [.2, .4, .6]])
        b2 = np.array([[.5, .5]]).transpose()

        theta3 = np.array([[.7, .9],
                           [.8, .1]])
        b3 = np.copy(b2)
        a1 = x
        z2 = np.dot(theta2, a1) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(theta3, a2) + b3
        a3 = sigmoid(z3)

        delta3 = (a3 - t) * sigmoid_derivative(z3)
        layer = Layer(weights_matrix=theta2, bias=b2, activation_function=sigmoid_activation.f,
                      activation_function_derivative=sigmoid_activation.derivative)
        cls.layer = layer
        (cls.x, cls.t, cls.theta2, cls.theta3,
         cls.b2, cls.b3, cls.z2, cls.z3,
         cls.a2, cls.a3, cls.delta3) = (x, t, theta2, theta3, b2, b3, z2, z3, a2, a3, delta3)

    def test_a_forward(self):
        expected_output = np.array([[.9866, .9950]]).transpose()
        layer, a1 = self.layer, self.x
        layer_output = layer.forward(previous_layer_output=a1, is_training=)
        self.assertTrue(np.allclose(layer_output, expected_output, rtol=0.0001))

    def test_b_backward(self):
        layer = self.layer
        theta3, delta3 = self.theta3, self.delta3
        delta = layer.backward(next_layer_weights=theta3, next_layer_delta=delta3)
        expected_delta = np.array([[0.00198391], [0.00040429]])
        self.assertTrue(np.allclose(delta, expected_delta, rtol=0.0001))

    def test_c_weight_update(self):
        expected_theta2 = np.array([[0.09998016, 0.29992064, 0.4999008],
                                    [0.19999596, 0.39998383, 0.59997979]])
        a = self.x
        layer = self.layer
        layer.weights_update(a, 0.01)
        new_theta2 = layer.theta
        self.assertTrue(np.allclose(expected_theta2, new_theta2))


class NNClassifierTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.array([[1, 4, 5]]).transpose()
        t = np.array([[.1, .05]]).transpose()
        alpha = 0.01
        sigmoid, sigmoid_derivative = sigmoid_activation
        theta2 = np.array([[.1, .3, .5],
                           [.2, .4, .6]])
        b2 = np.array([[.5, .5]]).transpose()

        theta3 = np.array([[.7, .9],
                           [.8, .1]])
        b3 = np.copy(b2)

        layer1 = Layer(weights_matrix=theta2, bias=b2, activation_function=sigmoid,
                       activation_function_derivative=sigmoid_derivative)
        layer2 = Layer(weights_matrix=theta3, bias=b3, activation_function=sigmoid,
                       activation_function_derivative=sigmoid_derivative)
        clf = NeuralNetworkClassifier([layer1, layer2], alpha, 'mse')
        cls.x, cls.t, cls.clf = x, t, clf

    def test_a_train(self):

        x, t, clf = self.x, self.t, self.clf
        clf.train(x=x.transpose(), y=t.transpose(), number_of_epochs=1)
        l2, l3 = clf.layers
        expected_new_theta2 = np.array([[.1, .2999, .4999],
                                        [.2, .4, .6]])
        expected_new_theta3 = np.array([[.6992, .8992],
                                        [.7988, .0988]])
        self.assertTrue(np.allclose(l2.theta, expected_new_theta2, atol=0.0001))
        self.assertTrue(np.allclose(l3.theta, expected_new_theta3, atol=0.0001))

    def test_validation(self):

        alpha = 0.01
        tanh, tanh_derivative = tanh_activation

        # layer 1 parameters
        theta1 = np.array([[4, 4],
                           [-3, -3]])
        b1 = np.array([[-2],
                       [5]])
        layer1 = Layer(weights_matrix=theta1, bias=b1, activation_function=tanh,
                       activation_function_derivative=tanh_derivative)

        # layer 2 parameters
        theta2 = np.array([[5, 5],
                           [5, 5]])
        b2 = np.array([[-5],
                      [-5]])
        layer2 = Layer(weights_matrix=theta2, bias=b2, activation_function=tanh,
                       activation_function_derivative=tanh_derivative)

        clf = NeuralNetworkClassifier([layer1, layer2], alpha, 'mse')

        # test 1 xor 1
        x = np.array([[0, 0]]).transpose()
        y = np.array([[-1, -1]])
        expected_mean_error = 0
        err = clf.validate(x, y)
        #self.assertTrue(expected_mean_error == clf.validate(x, y))
        self.assertTrue(np.allclose(err, expected_mean_error, atol=np.exp(-7)))

        # test 0 xor 0
        x = np.array([[1, 1]]).transpose()
        y = np.array([[-1, -1]])
        expected_mean_error = 0
        err = clf.validate(x, y)
        #self.assertTrue(expected_mean_error == clf.validate(x, y))
        self.assertTrue(np.allclose(err, expected_mean_error, atol=np.exp(-7)))

        # test 0 xor 1
        x = np.array([[0, 1]]).transpose()
        y = np.array([[1, 1]])
        expected_mean_error = 0
        err = clf.validate(x, y)
        #self.assertTrue(expected_mean_error == clf.validate(x, y))
        self.assertTrue(np.allclose(err, expected_mean_error, atol=np.exp(-7)))

        # test 1 xor 0
        x = np.array([[1, 0]]).transpose()
        y = np.array([[1, 1]])
        expected_mean_error = 0
        err = clf.validate(x, y)
        #self.assertTrue(expected_mean_error == clf.validate(x, y))
        self.assertTrue(np.allclose(err, expected_mean_error, atol=np.exp(-7)))


    def test_gradient(self):
        norm_x = np.random.ranf(3).reshape((3, 1))
        norm_y = np.random.ranf(2).reshape((2, 1))
        theta2 = np.random.randn(5, 3)
        theta3 = np.random.randn(3, 5)
        theta4 = np.random.randn(2, 3)
        b2 = np.random.randn(5, 1)
        b3 = np.random.randn(3, 1)
        b4 = np.random.randn(2, 1)
        l1 = Layer(weights_matrix=theta2, bias=b2, activation_function=sigmoid_activation.f,
                   activation_function_derivative=sigmoid_activation.derivative)
        l2 = Layer(weights_matrix=theta3, bias=b3, activation_function=sigmoid_activation.f,
                   activation_function_derivative=sigmoid_activation.derivative)
        l3 = Layer(weights_matrix=theta4, bias=b4, activation_function=sigmoid_activation.f,
                   activation_function_derivative=sigmoid_activation.derivative)
        # print(f'norm_x {norm_x}')
        clf = NeuralNetworkClassifier(layers=[l1, l2, l3], learning_rate=0.01, loss_function='mse')
        clf.check_gradient(norm_x.transpose(), norm_y.transpose())
        # self.clf.check_gradient(norm_x.transpose(), norm_y.transpose())


if __name__ == '__main__':
    unittest.main()