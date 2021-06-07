import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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
print('ML MSE:')
mse = q2.ml_evaluate(x_train, y_train, x_val, y_val)
print(mse)
print('MAP MSE:')
print(q2.map_evaluate(sol.x, x_train, y_train, x_val, y_val))

gamma = []
ml_mse = []
map_mse = []
for i in np.arange(0.0000001, 1, 0.001):
    gamma.append(i)
    ml_mse.append(mse)
    map_mse.append(q2.map_evaluate(i, x_train, y_train, x_val, y_val))
plt.plot(gamma, map_mse, color='blue')
plt.plot(gamma, ml_mse, color='red')
plt.title('MSE of MAP Estimator vs Gamma')
plt.legend(
        handles=[Line2D([0], [0], color='blue', label='MAP MSE', markersize=10),
                 Line2D([0], [0], color='red', label='ML MSE', markersize=10)])
plt.show()


