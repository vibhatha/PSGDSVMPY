import numpy as np


def test(e):
    X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],

    ])

    Y = np.array([-1, -1, 1, 1, 1])

    w = np.zeros(len(X[0]))
    eta = 1
    epochs = e

    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
            else:
                w = w + eta * (-2 * (1 / epoch) * w)






