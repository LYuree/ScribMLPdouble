import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


import csv

# X_train, y_train = load_mnist('data/mnist', kind='train')
X_train, y_train = None, None
X_test, y_test = None, None

with open('data/mnist_train.csv', 'r') as read_obj:
    train_set = np.array([list(map(int, rec)) for rec in csv.reader(read_obj, delimiter=',')])
    print(train_set.shape)
    y_train = train_set[:, 0]
    X_train = train_set[:, 1:]

with open('data/mnist_test.csv', 'r') as read_obj:
    test_set = np.array([list(map(int, rec)) for rec in csv.reader(read_obj, delimiter=',')])
    y_test = test_set[:, 0]
    X_test = test_set[:, 1:]

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

from scipy.special import expit
import sys


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
      Number of output units, should be equal to the
      number of unique class labels.

    n_features : int
      Number of features (dimensions) in the target dataset.
      Should be equal to the number of columns in the X array.

    n_hidden : int (default: 30)
      Number of hidden units.

    l1 : float (default: 0.0)
      Lambda value for L1-regularization.
      No regularization if l1=0.0 (default)

    l2 : float (default: 0.0)
      Lambda value for L2-regularization.
      No regularization if l2=0.0 (default)

    epochs : int (default: 500)
      Number of passes over the training set.

    eta : float (default: 0.001)
      Learning rate.

    alpha : float (default: 0.0)
      Momentum constant. Factor multiplied with the
      gradient of the previous epoch t-1 to improve
      learning speed
      w(t) := w(t) - (grad(t) + alpha*grad(t-1))

    decrease_const : float (default: 0.0)
      Decrease constant. Shrinks the learning rate
      after each epoch via eta / (1 + epoch*decrease_const)

    shuffle : bool (default: False)
      Shuffles training data every epoch if True to prevent circles.

    minibatches : int (default: 1)
      Divides training data into k minibatches for efficiency.
      Normal gradient descent learning if k=1 (default).

    random_state : int (default: None)
      Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features, n_hidden=(30),
                 l1=0.0, l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.__layers_count = len(n_hidden) + 2
        self.n_hidden = n_hidden
        self.weights = [None for l in range(self.__layers_count - 1)]
        self.a = [None for l in range(self.__layers_count)]
        self.z = [None for l in range(self.__layers_count-1)]
        self.sigma = [None for l in range(self.__layers_count-1)]
        self.grad = [None for l in range(self.__layers_count - 1)]
        self.delta_w = [None for l in range(self.__layers_count - 1)]
        self.delta_w_prev = [None for l in range(self.__layers_count - 1)]
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        # input weights
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden[0] * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden[0], self.n_features + 1)
        self.weights[0] = w1
        self.delta_w[0] = np.zeros(self.weights[0].shape)
        self.delta_w_prev[0] = np.zeros(self.weights[0].shape)

        # hidden weights
        for i in range(0, self.__layers_count - 3):
            wi = np.random.uniform(-1.0, 1.0, size=self.n_hidden[i+1]*(self.n_hidden[i] + 1))
            wi = wi.reshape(self.n_hidden[i+1], self.n_hidden[i] + 1)
            self.weights[i+1] = wi
            self.delta_w[i+1] = np.zeros(self.weights[i+1].shape)
            self.delta_w_prev[i+1] = np.zeros(self.weights[i+1].shape)

        # output weights
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden[-1] + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden[-1] + 1)
        self.weights[-1] = w2
        self.delta_w[-1] = np.zeros(self.weights[-1].shape)
        self.delta_w_prev[-1] = np.zeros(self.weights[-1].shape)
        # print(self.weights)
        # for item in self.weights:
        # print([len(a) for a in item])
        # print(len(item))
        return w1, w2

    # def _initialize_weights(self):
    #   """Initialize weights with small random numbers."""
    #   w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
    #   w1 = w1.reshape(self.n_hidden, self.n_features + 1)
    #   w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
    #   w2 = w2.reshape(self.n_output, self.n_hidden + 1)
    #   for item in self.weights:
    #     # print([len(a) for a in item])
    #     print(len(item))
    #   return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)
        # return np.abs(2*z) # this is VERY wrong, but I need it at the debugging stage
        # cause my VSCode and Pycharm compiler just throw errors with expit (lol)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        a1 : array, shape = [n_samples, n_features+1]
          Input values with bias unit.

        z2 : array, shape = [n_hidden, n_samples]
          Net input of hidden layer.

        a2 : array, shape = [n_hidden+1, n_samples]
          Activation of hidden layer.

        z3 : array, shape = [n_output_units, n_samples]
          Net input of output layer.

        a3 : array, shape = [n_output_units, n_samples]
          Activation of output layer.

        """
        # ai = self._add_bias_unit(X, how='column')
        # a1 = ai
        self.a[0] = self._add_bias_unit(X, how='column')
        self.z[0] = self.weights[0].dot(self.a[0].T)
        # self.z[0] = z2
        # a2 = self._sigmoid(z2)
        # a2 = self._add_bias_unit(a2, how='row')
        # zi = z2
        # ai = a1

        for i in range(1, self.__layers_count - 1):
            # ai = self._sigmoid(self.z[i-1])
            # ai = self._add_bias_unit(ai, how='row')
            # zi = self.weights[i].dot(ai)
            # z2 = zi
            self.a[i] = self._sigmoid(self.z[i-1])
            self.a[i] = self._add_bias_unit(self.a[i], how='row')
            self.z[i] = self.weights[i].dot(self.a[i])

        # a2 = self._sigmoid(z2)
        # a2 = self._add_bias_unit(a2, how='row')
        # a2 = self.a[2]
        # z3 = self.weights[-1].dot(self.a[2])
        # a3 = self._sigmoid(self.z[-1])
        # self.z[-1] = z3
        self.a[-1] = self._sigmoid(self.z[-1])
        # a3 = ai
        a1 = self.a[0]
        a2 = self.a[-2]
        a3 = self.a[-1]
        z2 = self.z[-2]
        z3 = self.z[-1]
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_):  # , w1, w2):
        """Compute L2-regularization cost"""
        # return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))
        sum = 0
        for wi in self.weights:
            sum += np.sum(wi[:, 1:] ** 2)
        return (lambda_ / 2.0) * sum

    def _L1_reg(self, lambda_):  # , w1, w2):
        """Compute L1-regularization cost"""
        # return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())
        sum = 0
        for wi in self.weights:
            sum += np.abs(wi[:, 1:]).sum()
        return (lambda_ / 2.0) * sum

    def _get_cost(self, y_enc, output):  # , w1, w2):
        """Compute cost function.

        y_enc : array, shape = (n_labels, n_samples)
          one-hot encoded class labels.

        output : array, shape = [n_output_units, n_samples]
          Activation of the output layer (feedforward)

        w1 : array, shape = [n_hidden_units, n_features]
          Weight matrix for input layer -> hidden layer.

        w2 : array, shape = [n_output_units, n_hidden_units]
          Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
          Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1)
        L2_term = self._L2_reg(self.l2)
        # L1_term = self._L1_reg(self.l1, w1, w2)
        # L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.

            Parameters
            ------------
            a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.

            a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.

            a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.

            z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.

            y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.

            w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.

            w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

            Returns
            ---------

            grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix w1.

            grad2 : array, shape = [n_output_units, n_hidden_units]
                Gradient of the weight matrix w2.

            """
        # backpropagation
        sigma3 = a3 - y_enc
        self.sigma[-1] = sigma3
        grad2 = sigma3.dot(self.a[-2].T)
        self.grad[-1] = grad2
        self.grad[-1][:, 1:] += (self.weights[-1][:, 1:] * (self.l1 + self.l2))
        self.delta_w[-1] = self.eta*self.grad[-1]
        self.weights[-1] -= (self.delta_w[-1] + (self.alpha * self.delta_w_prev[-1]))
        self.delta_w_prev[-1] = self.delta_w[-1]
        # self.z[-2] = z2 # do I need that?
        sigmai = sigma3
        # gradi = grad2
        for i in reversed(range(self.__layers_count - 3)):
            # sigmai is basically sigma[i+2] in terms of this loop
            # like, we have i = 0 => we take self.sigma[i+2], which is basically sigma3
            zi = self._add_bias_unit(self.z[i+1], how='row')
            sigmai = self.weights[i+2].T.dot(sigmai)*self._sigmoid_gradient(zi)
            sigmai = sigmai[1:, :]
            self.sigma[i+1] = sigmai
            # gradi = sigmai.dot(self.a[i+2].T)
            gradi = sigmai.dot(self.a[i+1].T)
            gradi[:, 1:] += (self.weights[i+1][:, 1:] * (self.l1 + self.l2))
            self.grad[i+1] = gradi
            self.delta_w[i+1] = self.eta*self.grad[i+1]
            self.weights[i+1] -= (self.delta_w[i+1] + (self.alpha * self.delta_w_prev[i+1]))
            self.delta_w_prev[i+1] = self.delta_w[i+1]

        zi = self._add_bias_unit(self.z[0], how='row')
        sigma2 = self.weights[1].T.dot(sigmai)*self._sigmoid_gradient(zi)
        sigma2 = sigma2[1:, :]
        self.sigma[0] = sigma2
        grad1 = sigma2.dot(self.a[0])
        self.grad[0] = grad1
        self.grad[0][:, 1:] += (self.weights[0][:, 1:] * (self.l1 + self.l2))
        self.delta_w[0] = self.eta*self.grad[0]
        self.weights[0] -= (self.delta_w[0] + (self.alpha * self.delta_w_prev[0]))
        self.delta_w_prev[0] = self.delta_w[0]
        # z2 = self._add_bias_unit(z2, how='row')
        # sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        # sigma2 = sigma2[1:, :]
        # self.sigma[0] = sigma2
        # grad1 = sigma2.dot(self.a[0])
        # self.grad[0] = grad1
        # self.grad[0][:, 1:] += (self.weights[0][:, 1:] * (self.l1 + self.l2))

        # sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        # sigma2 = sigma2[1:, :]
        # grad1 = sigma2.dot(a1)
        # grad2 = sigma3.dot(a2.T)

        # regularize
        # grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        # grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
          Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
          Input layer with original features.

        y : array, shape = [n_samples]
          Target class labels.

        print_progress : bool (default: False)
          Prints progress as the number of epochs
          to stderr.

        Returns:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w_w1_prev = np.zeros(self.w1.shape)
        delta_w_w2_prev = np.zeros(self.w2.shape)
        self.delta_w_prev[0] = delta_w_w1_prev
        self.delta_w_prev[-1] = delta_w_w2_prev

        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx],
                                      output=a3)
                # w1=self.w1,
                # w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1,
                                                  w2=self.w2)
                                                  


                # delta_w_w1, delta_w_w2 = self.eta * grad1, self.eta * grad2
                # self.w1 -= (delta_w_w1 + (self.alpha * delta_w_w1_prev))
                # self.w2 -= (delta_w_w2 + (self.alpha * delta_w_w2_prev))
                # delta_w_w1_prev, delta_w_w2_prev = delta_w_w1, delta_w_w2
                self.grad = [None for l in range(self.__layers_count - 1)]

        return self


nn = NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=[128],
                  l2=0.1,
                  l1=0.0,
                  epochs=45,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))
