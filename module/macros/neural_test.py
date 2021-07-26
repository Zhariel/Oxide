import matplotlib.pyplot as plt
from img_driver import Img_Driver
from datetime import datetime
import numpy as np
import random
import time
import sys
import os

driver = Img_Driver()

X = driver.pixels_rgb(16, rgb=False, divider=255)
Y = driver.define_y(16, rgb=False)

X_test = driver.pixels_rgb(16, rgb=False, divider=255, test=True)
Y_test = driver.define_y(16, rgb=False, test=True)

random.seed(0)
random.shuffle(X)
random.seed(0)
random.shuffle(Y)

print(f"X : {len(X)} {len(X[0])}")
print(X[0:1])
print()
print(f"Y : {len(Y)}")
print(Y[0:10])
print()

print(f"X_test : {len(X_test)} {len(X_test[0])}")
print(X_test[0:1])
print()
print(f"Y_test : {len(Y_test)}")
print(Y[0:10])

from ruster_neural import Ruster_Neural

r = Ruster_Neural(
    path="C:\\Users\\Revive\\PycharmProjects\\Rust_Native_ML\\rust\\PerceptronLib\\target\\release\\PerceptronLib.dll")
t1 = time.time()

XF = [item for sublist in X for item in sublist]
YF = [item for sublist in Y for item in sublist]

print(len(XF))

layer_count = 3
input_count = 256
hidden_count = 768
output_count = 3

losses = []
test_losses = []

nn = r.create_nn(layer_count, input_count, hidden_count, output_count)

value = r.predict_nn(nn, X[0], True, 3)
value = [value[0], value[1], value[2]]
print(value)
print()

for _ in range(1):
    r.train_nn(nn, XF, YF, 10, 0.003, True)

    rand_idx = random.randint(0, len(X) - 1)
    y_predict = r.predict_nn(nn, X[rand_idx], True, 3)
    y_predict = [y_predict[0], y_predict[1], y_predict[2]]
    loss = r.mse(y_predict, Y[rand_idx])
    losses.append(loss)

    dummy_y = r.predict_nn(nn, X[0], True, 3)
    dummy_predict = [dummy_y[0], dummy_y[1], dummy_y[2]]
    test_loss = r.mse(dummy_y, Y[0])
    test_losses.append(test_loss)

    if _ % 10 == 0:

        print(f"[{_}] [loss : {round(loss, 8)}] [idx: {rand_idx}] [predict: {y_predict}] [Y: {Y[rand_idx]}]")
        t2 = time.time()
        print(t2 - t1)
        print()

print()
plt.plot(losses)
plt.plot(test_losses)
plt.show()

r.store_nn(nn, 1111)
r.release_nn(nn)