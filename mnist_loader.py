from pickle import _Unpickler
import gzip

import numpy as np


def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        u = _Unpickler(f)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_solid_data():
    return np.load('data/solid data.npy', allow_pickle=True)


def load_cropped_data():
    x = np.load('data/x_data.npy', allow_pickle=True)
    y = np.load('data/y_data.npy', allow_pickle=True)
    return x, y
