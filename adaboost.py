"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]  # num of samples
        D = np.ones(m) / m
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_hat = self.h[t].predict(X)
            epsilon = np.sum(D * np.where(y_hat != y, 1, 0))
            self.w[t] = 0.5 * np.log((1 / epsilon) - 1)
            numerator = (D * np.exp(-self.w[t] * y * self.h[t].predict(X)))
            D = numerator / np.sum(numerator)
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        m = X.shape[0]  # num of samples
        y_hat_sum = np.zeros(m)
        for t in range(max_t):
            y_hat = self.w[t] * self.h[t].predict(X)
            y_hat_sum += y_hat
        y_hat_sum = np.where(y_hat_sum > 0, 1, -1)
        return y_hat_sum

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        m = X.shape[0]  # num of samples
        y_pred = self.predict(X, max_t)
        error = np.where(y_pred != y, 1, 0)
        return np.sum(error) / m
