import numpy as np
from sklearn.metrics import mean_squared_error


def cubic_z(x):
    n = x.shape[1]
    z = np.zeros(shape=(10, n))
    for i in range(n):
        z[0][i] = 1
        z[1][i] = x[0][i]
        z[2][i] = x[1][i]
        z[3][i] = x[0][i] ** 2
        z[4][i] = x[0][i] * x[1][i]
        z[5][i] = x[1][i] ** 2
        z[6][i] = x[0][i] ** 3
        z[7][i] = (x[0][i] ** 2) * x[1][i]
        z[8][i] = x[0][i] * (x[1][i] ** 2)
        z[9][i] = x[1][i] ** 3
    return z


def map_estimate(gamma, z, y):
    n = len(y)
    l = 1 / (n * gamma)
    q = np.zeros((10, 1))
    r = np.zeros((10, 10))
    for i in range(n):
        zi = np.array([z[:, i]]).T
        q = q + y[i] * zi
        r = r + np.matmul(zi, zi.T)
    q = q * (1 / n)
    r = r * (1 / n)
    w = np.matmul(np.linalg.inv((np.add(r, l * np.identity(10)))), q)
    return w


def ml_estimate(z, y):
    n = len(y)
    q = np.zeros((10, 1))
    r = np.zeros((10, 10))
    for i in range(n):
        zi = np.array([z[:, i]]).T
        q = q + y[i] * zi
        r = r + np.matmul(zi, zi.T)
    q = q * (1 / n)
    r = r * (1 / n)
    w = np.matmul(np.linalg.inv(r), q)
    return w


def map_evaluate(gamma, x_train, y_train, x_val, y_val):
    w = map_estimate(gamma, cubic_z(x_train), y_train)
    return mean_squared_error(y_val, np.matmul(w.T, cubic_z(x_val))[0, :])


def ml_evaluate(x_train, y_train, x_val, y_val):
    w = ml_estimate(cubic_z(x_train), y_train)
    return mean_squared_error(y_val, np.matmul(w.T, cubic_z(x_val))[0, :])
