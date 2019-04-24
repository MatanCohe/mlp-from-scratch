import unittest
import numpy as np

from NNClassifier import Layer
from functions import sigmoid_activation


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
        layer = Layer(weights_matrix=theta2, bias=b2,
                      activation_function=sigmoid_activation.f,
                      activation_function_derivative=sigmoid_activation.derivative,
                      learning_rate=alpha)
        cls.layer = layer
        (cls.x, cls.t, cls.theta2, cls.theta3,
         cls.b2, cls.b3, cls.z2, cls.z3,
         cls.a2, cls.a3, cls.delta3) = (x, t, theta2, theta3, b2, b3, z2, z3, a2, a3, delta3)

    def test_a_forward(self):
        expected_output = np.array([[.9866, .9950]]).transpose()
        layer, a1 = self.layer, self.x
        layer_output = layer.forward(previous_layer_output=a1)
        self.assertTrue(np.allclose(layer_output, expected_output, rtol=0.0001))

    def test_b_backward(self):
        layer = self.layer
        theta3, delta3 = self.theta3, self.delta3
        delta = layer.backward(next_layer_weights=theta3, next_layer_delta=delta3)
        expected_delta = np.array([[0.00198391], [0.00040429]])
        self.assertTrue(np.allclose(delta, expected_delta, rtol=0.0001))

    def test_c_weight_update(self):
        alpha = 0.01
        expected_theta2 = np.array([[0.09998016, 0.29992064, 0.4999008],
                                    [0.19999596, 0.39998383, 0.59997979]])
        a = self.x
        layer = self.layer
        layer.weights_update(a)
        new_theta2 = layer.theta
        self.assertTrue(np.allclose(expected_theta2, new_theta2))

if __name__ == '__main__':
    unittest.main()