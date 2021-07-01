from ctypes import *
import numpy as np
import matplotlib as plt

arr1 = np.array([1.0, 2.0, 3.0])
arr2 = np.array([10.0, 20.0, 30.0])


def test():
    print(1)


class Ruster():
    def __init__(self, path="dll/PerceptronLib.dll"):
        self.path = path
        self.lib = cdll.LoadLibrary(path)


    def npify(self, model, size):
        return np.ctypeslib.as_array(model, (size,))

    def destroy_linear_model(self, native_model, size):
        self.lib.destroy_model.argtypes = [POINTER(c_float), c_int]
        self.lib.destroy_model.restype = None
        self.lib.destroy_model(native_model, size)

    def create_nn(self, layer_count, input_len, hidden_len, output_len):
        self.lib.create_NeuralNet.argtypes = [c_int, c_int, c_int, c_int]
        self.lib.create_NeuralNet.restype = c_void_p

        return self.lib.create_NeuralNet(layer_count, input_len, hidden_len, output_len)

    def train_nn(self, neural_net, dataset, expected_outputs, epoch, alpha, isclassif):
        dataset_type = c_float * len(dataset)
        native_dataset = dataset_type(*dataset)

        expected_type = c_float * len(expected_outputs)
        expected_native = expected_type(*expected_outputs)

        self.lib.train_NeuralNet.argtypes = [c_void_p, POINTER(c_float), c_int, POINTER(c_float), c_int, c_int, c_float, c_bool]

        self.lib.train_NeuralNet(neural_net, native_dataset, len(dataset), expected_native, len(expected_outputs), epoch, alpha, isclassif)

    def predict_nn(self, neural_net, inputs, is_classif, output_size):
        input_type = c_float * len(inputs)

        self.lib.predict.argtypes = [c_void_p, POINTER(c_float), c_bool]
        self.lib.predict.restype = POINTER(c_float)

        native_input = input_type(*inputs)
        native_output = self.lib.predict(neural_net, native_input, is_classif)

        output = self.npify(native_output, output_size)
        self.destroy_linear_model(native_output, output_size)

        return output


    def release_nn(self, neural_net):
        self.lib.release_NeuralNet.argtypes = [c_void_p]
        self.lib.release_NeuralNet.restype = None

        self.lib.release_NeuralNet(neural_net)


