import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.special import expit

# Distribution parameters
priors = np.array([0.65, 0.35])
m01 = np.array([3, 0])
C01 = np.array([[2, 0], [0, 1]])
m02 = np.array([0, 3])
C02 = np.array([[1, 0], [0, 2]])
m1 = np.array([2, 2])
C1 = np.array([[1, 0], [0, 1]])


def generate_dataset(n):
    data = np.zeros(shape=(2, n))
    labels = [0 if random.uniform(0, 1) < 0.65 else 1 for i in range(n)]
    for i in range(n):
        if labels[i] == 0:
            if random.uniform(0, 1) < 0.5:
                data[:, i] = np.random.multivariate_normal(mean=m01, cov=C01).T
            else:
                data[:, i] = np.random.multivariate_normal(mean=m02, cov=C02).T
        else:
            data[:, i] = np.random.multivariate_normal(mean=m1, cov=C1).T
    return data, labels


def plot_dataset(data, labels, title):
    plt.scatter(data[0], data[1], color=['blue' if i == 0 else 'red' for i in labels])
    plt.title(title)
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='b', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='r', markersize=10)])
    plt.show()


def linear_z(x):
    n = x.shape[1]
    z = np.zeros(shape=(3, n))
    for i in range(n):
        z[0][i] = 1
        z[1][i] = x[0][i]
        z[2][i] = x[1][i]
    return z


def quadratic_z(x):
    n = x.shape[1]
    z = np.zeros(shape=(6, n))
    for i in range(n):
        z[0][i] = 1
        z[1][i] = x[0][i]
        z[2][i] = x[1][i]
        z[3][i] = x[0][i] ** 2
        z[4][i] = x[0][i] * x[1][i]
        z[5][i] = x[1][i] ** 2
    return z


def linear_logistic_function(x, w):
    wTz = np.matmul(w.T, np.array([[1, x[0], x[1]]]).T)
    return expit(wTz)


def quadratic_logistic_function(x, w):
    wTz = np.matmul(w.T, np.array([[1, x[0], x[1], x[0] * x[0], x[0] * x[1], x[1] * x[1]]]).T)
    return expit(wTz)


def cost(w, x, l, h):
    n = x.shape[1]
    c = 0
    for i in range(n):
        c = c + l[i] * np.log(h(x[:, i], w)) + (1 - l[i]) * np.log(1 - h(x[:, i], w))
    return c * (-1/n)


def logistic_regression(train_x, train_l, val_x, val_l, title, quadratic=False):
    if quadratic:
        h = quadratic_logistic_function
        z = quadratic_z
        w = np.array([[0, 0, 0, 0, 0, 0]]).T
    else:
        h = linear_logistic_function
        z = linear_z
        w = np.array([[0, 0, 0]]).T

    sol = minimize(cost, w, args=(train_x, train_l, h))
    w = sol.x
    d = np.matmul(w.T, z(val_x))
    dL0 = d[[i == 0 for i in val_l]]
    dL1 = d[[i == 1 for i in val_l]]
    p_error = (np.count_nonzero(dL0 >= 0) + np.count_nonzero(dL1 < 0)) / np.size(d)
    print(title + ' Pr(error):')
    print(p_error)
    L0 = val_x[:, [i == 0 for i in val_l]]
    L1 = val_x[:, [i == 1 for i in val_l]]
    colors0 = ['green' if dL0[i] < 0 else 'red' for i in range(len(dL0))]
    colors1 = ['green' if dL1[i] >= 0 else 'red' for i in range(len(dL1))]
    plt.scatter(L0[0], L0[1], marker='o', color=colors0)
    plt.scatter(L1[0], L1[1], marker='^', color=colors1)
    plt.title(title)
    plt.show()
