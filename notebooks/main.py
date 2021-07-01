from ruster import Ruster
import numpy as np

def unpack(lis):
    flatlist = []

    for elt in lis:
        for sub_elt in elt:
            flatlist.append(sub_elt)

    return flatlist

if __name__ == '__main__':
    r = Ruster(path="C:\\Users\\Revive\\PycharmProjects\\Rust_Native_ML\\rust\\PerceptronLib\\target\\debug\\PerceptronLib.dll")

    size = 2
    inp_size = 6
    inputs = [
        [1, 4],
        [1, -4],
        [4, 4],
    ]
    y = [
        1,
        1,
        -1
    ]
    expected_outputs = []

    # native_model = r.create_linear_model(size)
    # model = r.npify(native_model, size+1)
    # print(model)
    # result = r.predict_linear_model_classification(native_model, [1, 3], size+1)
    # print(result)

    # r.train_rosenblatt_linear_model(model, inputs, expected_outputs, size, inp_size, iterations=20, alpha=0.001)

    # r.destroy_linear_model(native_model, size)

    layer_count = 2
    input_count = 2
    hidden_count = 3
    output_count = 1

    print(r.sum([1, 2, 3]))

    nn = r.create_nn(layer_count, input_count, hidden_count, output_count)

    r.train_nn(nn, unpack(inputs), y, 100, 0.03, True)

    i = 1
    j = 2

    print(r.predict_nn(nn, [1, 4], True, output_count))

    r.release_nn(nn)