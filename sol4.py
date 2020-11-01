from ex4_tools import *
import numpy as np
import adaboost as adb

NUM_SAMPLES_TRAIN = 5000
NUM_SAMPLES_TEST = 200
NO_NOISE = 0
NOISE_RATIO = [0.01, 0.04]
T = 500
T_LIST = [5, 10, 50, 100, 200, 500]


def calc_train_and_test_err(x_train, y_train, noise):
    """
    This function calculate the train and test error vectors
    :param x_train: the samples of the train data to calculate its error
    :param y_train: the labels of the train data to calculate its error
    :param noise: the noise to generate data with (the test data)
    :return: the train and test error vectors
    """
    x_test, y_test = generate_data(NUM_SAMPLES_TEST, noise)
    boost = adb.AdaBoost(DecisionStump, T)
    boost.train(x_train, y_train)

    train_err = []
    test_err = []
    for t in range(T):
        train_err.append(boost.error(x_train, y_train, t + 1))
        test_err.append(boost.error(x_test, y_test, t + 1))
    return train_err, test_err


def q_10(noise):
    """
    This function operates the required in q_10- plots the train and test error
    :param noise: the noise to generate data with (the test data)
    """
    x_train, y_train = generate_data(NUM_SAMPLES_TRAIN, noise)
    train_err, test_err = calc_train_and_test_err(x_train, y_train, noise)

    # The plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("train & test error vs. T")
    plt.plot(np.arange(T), train_err, label="Train set error")
    plt.plot(np.arange(T), test_err, label="Test set error")
    plt.legend()
    ax.set_xlabel('T')
    ax.set_ylabel('error')
    fig.show()


def q_11(noise):
    """
    This function operates the required in q_11- plots the decisions of the learned
    classifier with T=[5, 10, 50, 100, 200, 500]
    :param noise: the noise to generate data with (the test data)
    """
    x_train, y_train = generate_data(NUM_SAMPLES_TRAIN, noise)
    x_test, y_test = generate_data(NUM_SAMPLES_TEST, noise)
    for i in range(len(T_LIST)):
        boost = adb.AdaBoost(DecisionStump, T_LIST[i])
        boost.train(x_train, y_train)
        plt.subplot(2, 3, i + 1)
        decision_boundaries(boost, x_test, y_test, T_LIST[i])
    plt.show()


def q_12(noise):
    """
    This function operates the required in q_12- finds the T that minimizes the test
    error, and plots the decisions boundaries of this classifier with the training data
    :param noise: the noise to generate data with (the test data)
    """
    x_train, y_train = generate_data(NUM_SAMPLES_TRAIN, noise)
    train_err, test_err = calc_train_and_test_err(x_train, y_train, noise)
    T_hat = np.argmin(test_err)
    min_boost = adb.AdaBoost(DecisionStump, T_hat)
    min_boost.train(x_train, y_train)
    decision_boundaries(min_boost, x_train, y_train, T_hat)
    plt.show()
    print("The T_hat is: " + str(T_hat) + " and its test error is: " + str(
        test_err[T_hat]))


def q_13(noise):
    """
    This function operates the required in q_13- plots the training set with size
    proportional to its weight in D^T. (after seeing the results with the original D^T,
    the function normalized the D^T and then plotted)
    :param noise: the noise to generate data with (the test data)
    """
    x_train, y_train = generate_data(NUM_SAMPLES_TRAIN, noise)

    # using D_T without normalize - cant see any points:
    # boost = adb.AdaBoost(DecisionStump, T)
    # D_T = boost.train(x_train, y_train)
    # decision_boundaries(boost, x_train, y_train, T, D_T)
    # plt.show()

    # using D_T with normalize:
    boost = adb.AdaBoost(DecisionStump, T)
    D_T = boost.train(x_train, y_train)
    D_T = (D_T / np.max(D_T)) * 10
    decision_boundaries(boost, x_train, y_train, T, D_T)
    plt.show()


def q_14():
    """
    This function operates the required in q_14- repeats questions: 10,11,12,13 with
    noised data
    """
    for noise in NOISE_RATIO:
        q_10(noise)
        q_11(noise)
        q_12(noise)
        q_13(noise)
