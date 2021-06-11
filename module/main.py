from ruster import Ruster

if __name__ == '__main__':
    r = Ruster()

    size = 2
    inp_size = 6
    inputs = [
        [1, 4],
        [1, -4],
        [4, 4],
    ]
    dataset_expected_outputs = [
        1,
        1,
        -1
    ]
    expected_outputs = []

    native_model = r.create_linear_model(size)
    model = r.npify_native_model(native_model, size+1)
    print(model)
    result  = r.predict_linear_model_classification(native_model, [1, 3], size+1)
    print(result)

    # r.train_rosenblatt_linear_model(model, inputs, expected_outputs, size, inp_size, iterations=20, alpha=0.001)

    r.destroy_linear_model(native_model, size)

    # print(r.sum([1, 2, 3]))