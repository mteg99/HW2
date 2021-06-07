import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from hw2q2 import hw2q2
import q2_api as q2

# Generate datasets
x_train, y_train, x_val, y_val = hw2q2()

# Plot datasets
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(x_train[0], x_train[1], y_train)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(x_val[0], x_val[1], y_val)
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()

sol = minimize(q2.map_evaluate, np.array(0.000001), args=(x_train, y_train, x_val, y_val))
print('Min Gamma:')
print(sol.x)
print('MAP MSE:')
print(q2.map_evaluate(sol.x, x_train, y_train, x_val, y_val))

gamma = []
mse = []
for i in np.arange(0.0000001, 0.0001, 0.0000001):
    gamma.append(i)
    w = q2.map_estimate(i, q2.cubic_z(x_train), y_train)
    mse.append(mean_squared_error(y_val, np.matmul(w.T, q2.cubic_z(x_val))[0, :]))
plt.plot(gamma, mse)
plt.show()

print('ML MSE:')
print(q2.ml_evaluate(x_train, y_train, x_val, y_val))
