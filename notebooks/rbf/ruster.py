from ctypes import *
import numpy as np
import matplotlib as plt


def test():
    print(1)


class Ruster():
    def __init__(self, path="dll/PerceptronLib.dll"):
        self.path = path
        self.lib = cdll.LoadLibrary(path)

    def sum(self, arr):
        arr_type = c_int * len(arr)
        self.lib.sum.argtypes = [arr_type, c_int]
        self.lib.sum.restype = c_int
        native_arr = arr_type(*arr)

        return self.lib.sum(native_arr, len(arr))

    def create_linear_model(self, size):
        self.lib.create_linear_model.argtypes = [c_int]
        self.lib.create_linear_model.restype = POINTER(c_float)

        return self.lib.create_linear_model(size)

    def npify(self, model, size):
        return np.ctypeslib.as_array(model, (size,))

    def destroy(self, native_model, size):
        self.lib.destroy_model.argtypes = [POINTER(c_float), c_int]
        self.lib.destroy_model.restype = None
        self.lib.destroy_model(native_model, size)

    def init_clusters(self, inputs, nb_clusters, ndim):
        input_type = c_float * len(inputs)
        inputs_native = input_type(*inputs)

        self.lib.init_clusters.argtypes = [POINTER(c_float), c_int, c_int, c_int]
        self.lib.init_clusters.restype = POINTER(c_float)

        return self.lib.init_clusters(inputs_native, len(inputs), nb_clusters, ndim)

    def k_means(self, inputs, clusters_raw, nb_clusters, ndim):
        input_type = c_float * len(inputs)
        inputs_native = input_type(*inputs)

        self.lib.k_means.argtypes = [POINTER(c_float), c_int, POINTER(c_float), c_int, c_int]

        self.lib.k_means(inputs_native, len(inputs), clusters_raw, nb_clusters, ndim)

    def predict_rbf_naive(self, model_raw, x, sample, ndim, gamma, classif):
        x_type = c_float * len(x)
        x_native = x_type(*x)

        sample_type = c_float * len(sample)
        sample_native = sample_type(*sample)

        self.lib.predict_rbf_classification.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_float]
        self.lib.predict_rbf_classification.restype = c_float

        self.lib.predict_rbf_regression.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_float]
        self.lib.predict_rbf_regression.restype = c_float

        if classif:
            return self.lib.predict_rbf_classification(model_raw, x_native, sample_native, len(x), ndim, gamma)
        else:
            return self.lib.predict_rbf_regression(model_raw, x_native, sample_native, len(x), ndim, gamma)

    def train_rosenblatt_rbf(self, model_raw, x, y, ndim, iterations, alpha, gamma):
        x_type = c_float * len(x)
        x_native = x_type(*x)

        y_type = c_float * len(y)
        y_native = y_type(*y)

        self.lib.train_rosenblatt_rbf.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_float, c_float]

        self.lib.train_rosenblatt_rbf(model_raw, x_native, y_native, len(x), ndim, iterations, alpha, gamma)