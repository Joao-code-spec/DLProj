#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))
        self.W_0 = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        sign = np.argmax(np.sign(np.dot(self.W, x_i.T)))
        if sign != y_i:
            self.W[y_i, :] = self.W[y_i, :] + x_i
            self.W[sign, :] = self.W[y_i, :] - x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        # """

        self.W = self.W_0 - learning_rate * (
                np.log(np.sum(np.exp(np.dot(self.W, x_i.T)))) - np.dot(self.W[y_i, :], x_i.T))
        # Q1.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size=200):
        self.W_1 = np.random.normal(loc=0.0, scale=1.0, size=(n_features, hidden_size))
        self.W_2 = np.random.normal(scale=0.1, size=(hidden_size, n_classes))
        self.bias_1 = np.zeros(hidden_size)
        self.bias_2 = np.zeros(n_classes)

    def softmax(self, x):
        return (np.exp(x) / np.exp(x).sum())

    def predict(self, X):
        A = np.dot(X, self.W_1) + self.bias_1
        A[A < 0] = 0

        B = np.dot(A, self.W_2) + self.bias_2
        B[B < 0] = 0

        predicted_labels = B.argmax(axis=0)

        # intermediate = np.dot(self.W_1, X.T)  # (n_classes x n_examples)
        # intermediate[intermediate < 0] = 0
        # scores = np.dot(self.W_2, intermediate.T)
        # predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def relu_deriv(self, x):
        return np.where(x < 0, 0, 1)

    def train_epoch(self, X, y, learning_rate=0.001):
        A = np.dot(X, self.W_1) + self.bias_1
        C = self.relu(A)
        B = np.dot(C, self.W_2) + self.bias_2
        D = self.relu(B)
        G = self.softmax(D)

        Eout = G - y
        Ehid = self.relu_deriv(A) * np.dot(Eout, self.W_2.T)
        dOut = np.outer(C, Eout)
        dHid = np.outer(X, Ehid)

        self.W_1 -= learning_rate * dHid
        print('finish')

        self.W_2 -= learning_rate * dOut
        print('finish')

        self.bias_1 -= learning_rate * Ehid
        print('finish')

        self.bias_2 -= learning_rate * Eout
        print('finish')

def plot(epochs, valid_accs, test_accs, name=''):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        print('aa')
        train_order = np.random.permutation(train_X.shape[0])
        print('aa')

        train_X = train_X[train_order]
        print('aa')

        train_y = train_y[train_order]
        print('aa')

        if opt.model == 'mlp':
            print('aa')

            b = np.zeros((train_y.size, train_y.max() + 1))
            print('aa')

            b[np.arange(train_y.size), train_y] = 1
            print('aa')

            train_y = b
        print('aa')

        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        print('aa')

        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs, f"./plots/q1_{opt.model}/accuracy")


if __name__ == '__main__':
    main()
