"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        total_accurancy_series = []
        total_precision_series = []
        total_recal_series = []
       
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                total_accuracy, accuracy_by_class, precision, recall = self.calculate_metrics(test_data)
                total_precision = np.mean(list(precision.values()))
                total_recall = np.mean(list(recall.values()))
                
                total_accurancy_series += [total_accuracy]
                total_precision_series += [total_precision]
                total_recal_series += [total_recall]
                
                print(f"Epoch {j}:")
                print("    Accuracy by Class:")
                for cls, accuracy in accuracy_by_class.items():
                    print(f"        Class {cls}: {accuracy}")
                print("    Precision:")
                for cls, precision_value in precision.items():
                    print(f"        Class {cls}: {precision_value}")
                print("    Recall:")
                for cls, recall_value in recall.items():
                    print(f"        Class {cls}: {recall_value}")                
                print(f"    Total Accuracy: {total_accuracy}")
                print(f"    Total Precision: {np.mean(list(precision.values()))}")
                print(f"    Total Recall: {np.mean(list(recall.values()))}")
            else:
                print("Epoch {0} complete".format(j))        
        
        return np.array(total_accurancy_series), np.array(total_precision_series), np.array(total_recal_series)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return a list of tuples containing the predicted and actual
        class labels for each test input in the given test data.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return test_results

    def calculate_metrics(self, test_data):
        """
        Calculate accuracy by class, precision, and recall for the given test data.
        """
        results = self.evaluate(test_data)
        class_counts = {}
        class_correct = {}
        true_positives = {}
        false_positives = {}
        false_negatives = {}

        for predicted, actual in results:
            if actual not in class_counts:
                class_counts[actual] = 0
                class_correct[actual] = 0
                true_positives[actual] = 0
                false_positives[actual] = 0
                false_negatives[actual] = 0
            if predicted not in false_positives:
                false_positives[predicted] = 0

            class_counts[actual] += 1
            if predicted == actual:
                class_correct[actual] += 1
                true_positives[actual] += 1
            else:
                false_positives[predicted] += 1
                false_negatives[actual] += 1

        total_accuracy = sum(predicted == actual for predicted, actual in results) / len(results)

        accuracy_by_class = {
            cls: class_correct[cls] / class_counts[cls] if class_counts[cls] > 0 else 0.0
            for cls in class_counts
        }

        precision = {
            cls: true_positives[cls] / (true_positives[cls] + false_positives[cls])
            if (true_positives[cls] + false_positives[cls]) > 0
            else 0.0
            for cls in class_counts
        }

        recall = {
            cls: true_positives[cls] / (true_positives[cls] + false_negatives[cls])
            if (true_positives[cls] + false_negatives[cls]) > 0
            else 0.0
            for cls in class_counts
        }

        return total_accuracy, accuracy_by_class, precision, recall

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
