import matplotlib.pyplot as plt
import numpy as np
from ruster import Ruster
import time
import sys
import os

color_1 = "salmon"
color_2 = "deepskyblue"
color_3 = "lightgreen"

r = Ruster(path="C:\\Users\\Revive\\PycharmProjects\\Rust_Native_ML\\rust\\PerceptronLib\\target\\release\\PerceptronLib.dll")

print(r.sum([1, 2, 3]))

X = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])
Y = np.array([
    1,
    -1,
    -1
])
#
x = X.flatten().tolist()

rbf = r.create_rbf(x, 2, 0.01)

r.release_rbf(rbf)