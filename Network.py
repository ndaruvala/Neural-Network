import numpy as np
import random
import Data

"""
Sigmoid function declarations.
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    y = 1 / (1 + np.exp(-x))
    return y * (1 - y)


class Network:
    """
    Initializes weights, biases, and sigmoid functions.
    """

    def __init__(self, layers):
        self.weights, self.biases = [], []
        for l in range(1, len(layers)):
            self.weights.append(
                np.random.randn(layers[l], layers[l - 1])
            )
            self.biases.append(np.zeros((layers[l], 1)))
        self.layers = layers
        self.sigmoid = np.vectorize(sigmoid)
        self.sigmoid_p = np.vectorize(sigmoid_p)

    """
    Feeds an input numpy array forward through the network.
    """

    def feedForward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    """
    Calculates activations and z for each layer.
    """

    def get_Z_A(self, a):
        Z = []
        A = []
        z = None
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
            Z.append(z)
            A.append(a)
        return Z, A

    '''
    Propgates backwards to calculate the change in the weights and baises 
    for one example.
    '''

    def backpropagationHelper(self, example):
        delta_dCdW = [np.zeros(w.shape) for w in self.weights]
        delta_dCdB = [np.zeros(b.shape) for b in self.biases]
        (a, y) = example
        Z, A = self.get_Z_A(a)
        delta = (A[-1] - y) * self.sigmoid_p(Z[-1])
        delta_dCdB[-1] = delta
        delta_dCdW[-1] = np.dot(delta, np.transpose(A[-2]))
        # Backproping
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(np.transpose(
                self.weights[i+1]), delta) * self.sigmoid_p(Z[i])
            delta_dCdB[i] += delta
            if i == 0:
                delta_dCdW[i] += np.dot(delta, np.transpose(a))
            else:
                delta_dCdW[i] += np.dot(delta, np.transpose(A[i - 1]))
        return delta_dCdW, delta_dCdB

    """
    Runs the backpropagation algorithm over a given mini_batch of data and adjusts 
    weights and baises accordingly.
    """

    def backpropagation(self, mini_batch):
        # Initialize nudges as zero
        dCdW, dCdB = [], []
        for l in range(1, len(self.layers)):
            dCdW.append(np.zeros((self.layers[l], self.layers[l - 1])))
            dCdB.append(np.zeros((self.layers[l], 1)))
        # Sets a default learning rate temporarily
        learn_rate = -3.0
        for example in mini_batch:
            delta_dCdW, delta_dCdB = self.backpropagationHelper(example)
            for i in range(len(self.weights)):
                dCdW[i] += delta_dCdW[i]
                dCdB[i] += delta_dCdB[i]
        # Take the average of the weights
        for dW, dB in zip(dCdW, dCdB):
            dW *= learn_rate / len(mini_batch)
            dB *= learn_rate / len(mini_batch)
        # Apply the nudges to the weights and biases of the network
        for i in range(len(self.weights)):
            self.weights[i] += dCdW[i]
            self.biases[i] += dCdB[i]

    """
    Evaluates the neural network on a set of training data.
    """

    def getAccuracy(self, test_data):
        correct = 0
        for x, y in test_data:
            if np.argmax(self.feedForward(x)) == y:
                correct += 1
        return (correct / len(test_data)) * 100

    """
    Splits the training data into mini batches. 
    For each mini batch, backpropagates across all examples.
    Tests the network if given test_data.
    Repeats until we run out of training data.

    input: 
        training_data: (x, y) tuples of input and expected ouput
        mini_batch_size: size of each mini batch
    """

    def gradientDescent(self, training_data, mini_batch_size, test_data, epochs):
        training_data = list(training_data)
        test_data = list(test_data)

        # Randomly shuffle data and split into batches
        for e in range(epochs):
            random.shuffle(training_data)
            n = len(training_data)
            mini_batches = [training_data[i: i + mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.backpropagation(mini_batch)
            accuracy = self.getAccuracy(test_data)
            print(f"Epoch {e} completed. Accuracy: {accuracy}%")


if __name__ == "__main__":
    (training_data, validation_data, test_data) = Data.load_data_wrapper()
    myNetwork = Network([784, 30, 10])
    myNetwork.gradientDescent(training_data, 10, test_data, 30)
