import numpy as np
import nnfs # sets random seed and dot product default datatype to ensure outcomes are repeatable
from nnfs.datasets import spiral_data

nnfs.init()

# np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # already flipped, no need for transpose
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: # rectified linear unit
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: # activation function for the output neurons
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(100, 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output[:5])

print(activation2.output)

# print(X)
# print(y)
# layer1 = Layer_Dense(2, 5)
# activation1 = Activation_ReLU()
# layer1.forward(X)
# activation1.forward(layer1.output)
# print(layer1.output)
# print(activation1.output)
