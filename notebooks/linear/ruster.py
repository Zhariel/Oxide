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

    def destroy_linear_model(self, native_model, size):
        self.lib.destroy_model.argtypes = [POINTER(c_float), c_int]
        self.lib.destroy_model.restype = None
        self.lib.destroy_model(native_model, size)

    def predict_linear_model_regression(self, model_raw, input, model_size):
        input_type = c_float * len(input)

        self.lib.predict_linear_model_regression.argtypes = [POINTER(c_float), input_type, c_int]
        self.lib.predict_linear_model_regression.restype = c_float

        inputs_native = input_type(*input)

        return self.lib.predict_linear_model_regression(model_raw, inputs_native, model_size)

    def predict_linear_model_classification(self, model_raw, input, model_size):
        input_type = c_float * len(input)

        self.lib.predict_linear_model_classification.argtypes = [POINTER(c_float), input_type, c_int]
        self.lib.predict_linear_model_classification.restype = c_float

        inputs_native = input_type(*input)

        return self.lib.predict_linear_model_classification(model_raw, inputs_native, model_size)

    def train_rosenblatt_linear_model(self, model_raw, input, expected_outputs, model_size, iterations, alpha):
        input_type = c_float * len(input)
        expected_type = c_float * len(expected_outputs)

        self.lib.train_rosenblatt_linear_model.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int,
                                                           c_int, c_int, c_float]

        inputs_native = input_type(*input)
        expected_native = expected_type(*expected_outputs)

        self.lib.train_rosenblatt_linear_model(model_raw, inputs_native, expected_native, model_size, len(input),
                                               iterations, alpha)

    def store_model(self, neural_net, size, file_nb):
        self.lib.store_model.argtypes = [POINTER(c_float), c_int, c_int]

        self.lib.store_model(neural_net, size, file_nb)

    def load_model(self, file_nb):
        self.lib.load_model.argtypes = [c_int]
        self.lib.load_model.restype = POINTER(c_float)

        return self.lib.load_model(file_nb)