import numpy as np
import random
import Data
import sys
import numpy

# numpy.set_printoptions(threshold=sys.maxsize)

"""
Sigmoid function declarations
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
        # layers : list of [input size, hidden size, hidden size, output size]
        # in terms of neuros
        # for each size in [1:], initialize weight size of current layer x size of prev layer
        # self.biases = [
        #     np.random.normal(size=(16, 1)),
        #     np.random.normal(size=(10, 1))
        # ]
        # self.weights = [
        #     np.random.normal(size=(16, 784)),
        #     np.random.normal(size=(10, 16)),
        # ]
        self.weights, self.biases = [], []
        for l in range(1, len(layers)):
            self.weights.append(
                np.random.normal(size=(layers[l], layers[l - 1]))
            )
            self.biases.append(np.random.normal(size=(layers[l], 1)))
        self.layers = layers
        self.sigmoid = np.vectorize(sigmoid)
        self.sigmoid_p = np.vectorize(sigmoid_p)

    """
    Feeds an input numpy array forward through the network.
    input: a : numpy array input data of 784 pixels.
    output: numpy array of 10 neurons which correspond to digits 0 - 9.
    """

    def feedForward(self, a):
        z = None
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return (a, z)

    """
    Calculates activations and z for each layer
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

    """
    Returns a tuple containing an array of nudge arrays for weights 
    an array of nudge arrays for biases for each layer.
    """

    def backpropagationHelper(self, a, y):
        # Initialize values
        Z, A = self.get_Z_A(a)
        weights, biases = [], []
        for l in range(1, len(self.layers)):
            weights.append(np.zeros((self.layers[l], self.layers[l - 1])))
            biases.append(np.zeros((self.layers[l], 1)))
        # Calculate values and fill biases/weights for output layer
        delta = (A[-1] - y) * self.sigmoid_p(Z[-1])
        biases[-1] = np.array([delta[j] for j in range(len(delta))])
        weights_T = np.transpose(self.weights[-1])
        rows, cols = np.shape(weights[-1])
        for j in range(rows):
            for k in range(cols):
                a_k_prev = A[-2][k]
                weights[-1][j, k] = a_k_prev * delta[j]
        # Backproping
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(weights_T, delta) * self.sigmoid_p(Z[i])
            weights_T = np.transpose(self.weights[i])
            biases[i] = np.array([delta[j] for j in range(len(biases[i]))])
            rows, cols = np.shape(weights[i])
            for j in range(rows):
                for k in range(cols):
                    if i == 0:
                        a_k_prev = a[k]
                    else:
                        a_k_prev = A[i - 1][k]
                    weights[i][j, k] = a_k_prev * delta[j]
        # print("dW for one example", weights[0], "separate", weights[1])
        # print("dB for one example", biases[0], "separate", biases[1])
        return weights, biases

    """
    Runs the backpropagation algorithm over a given mini_batch of data
    """

    def backpropagation(self, mini_batch):
        # Initialize nudges as zero
        dCdW, dCdB = [], []
        for l in range(1, len(self.layers)):
            dCdW.append(np.zeros((self.layers[l], self.layers[l - 1])))
            dCdB.append(np.zeros((self.layers[l], 1)))
        # Sets a default learning rate temporarily
        learn_rate = -3.0
        # Sums nudges over all examples
        for example in mini_batch:
            (a, y) = example
            dWeights, dBiases = self.backpropagationHelper(a, y)
            for i in range(len(dCdW)):
                dCdW[i] += dWeights[i]
                dCdB[i] += dBiases[i]
        # print("Summed nudges over all examples.")
        # Take the average of the weights
        for weights, biases in zip(dCdW, dCdB):
            weights *= learn_rate / len(mini_batch)
            biases *= learn_rate / len(mini_batch)
        # print("Averaged nudges over all examples.")
        # Apply the nudges to the weights and biases of the network
        for i in range(len(self.weights)):
            # print("dCdW", dCdW[i])
            self.weights[i] += dCdW[i]
            # print("dCdB", dCdB[i])
            self.biases[i] += dCdB[i]

    # print("Adjusted weights and biases.")

    """
    Splits the training data into mini batches. 
    For each mini batch, updates all weights and biases by backpropagating 
    across all examples in the given mini batch.
    Tests the network across given test_data.
    Repeats until we run out of training data.

    input: 
        training_data: (x, y) tuples of input and expected ouput
        mini_batch_size: size of each mini batch

    output:
        Nothing. Adjusts all weights and biases in the Network appropriately.
    """

    def gradientDescent(self, training_data, mini_batch_size, test_data):
        training_data = list(training_data)
        test_data = list(test_data)

        # Randomly shuffle data and split into batches
        random.shuffle(training_data)
        mini_batches = []
        for i in range(
            0, len(training_data) - mini_batch_size, mini_batch_size
        ):
            mini_batches.append(training_data[i : i + mini_batch_size])
        for mini_batch in mini_batches:
            self.backpropagation(mini_batch)
            # print("Backpropagation on mini_batch executed.")
            correct = 0
            for x, y in test_data:
                if np.argmax(self.feedForward(x)) == y:
                    correct += 1
            accuracy = (correct / len(test_data)) * 100
            print(f"Mini-Batch Completed: Accuracy: {accuracy}%")


if __name__ == "__main__":
    (training_data, validation_data, test_data) = Data.load_data_wrapper()
    myNetwork = Network([784, 16, 16, 10])
    myNetwork.gradientDescent(training_data, 64, test_data)


# """
# Returns the sums of the square of the differences between expected vector
#  and observed vector for a single example
# """

# def quadraticCost(self, x, y):
#     a = self.feedForward(x)[0]
#     n = len(y)
#     sum = 0
#     for i in range(n):
#         sum += (y[i] - a[i]) ** 2
#     return (1 / 2) * sum
