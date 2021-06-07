import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import q1_api as q1

# Generate datasets
D20_train, L20_train = q1.generate_dataset(20)
D200_train, L200_train = q1.generate_dataset(200)
D2000_train, L2000_train = q1.generate_dataset(2000)
D10k_val, L10k_val = q1.generate_dataset(10000)

# Plot datasets
# q1.plot_dataset(D20_train, L20_train, '20 Training Samples')
# q1.plot_dataset(D200_train, L200_train, '200 Training Samples')
# q1.plot_dataset(D2000_train, L2000_train, '2000 Training Samples')
q1.plot_dataset(D10k_val, L10k_val, '10k Validation Samples')

# Optimal classifier
FPR = []
TPR = []
gammas = []
p_error = []
pxL0 = 0.5*stats.multivariate_normal.pdf(D10k_val.T, q1.m01, q1.C01) + \
       0.5*stats.multivariate_normal.pdf(D10k_val.T, q1.m02, q1.C02)
pxL1 = stats.multivariate_normal.pdf(D10k_val.T, q1.m1, q1.C1)
d = np.divide(pxL1, pxL0)
dL0 = d[[i == 0 for i in L10k_val]]
dL1 = d[[i == 1 for i in L10k_val]]
gamma_min = 0
gamma_max = np.max(d)
gamma_inc = 0.1
gamma = gamma_min
while gamma < gamma_max:
    FP = np.count_nonzero(dL0 >= gamma)
    N = np.size(dL0)
    FPR.append(FP / N)
    TP = np.count_nonzero(dL1 >= gamma)
    P = np.size(dL1)
    TPR.append(TP / P)
    FN = np.count_nonzero(dL1 < gamma)
    p_error.append((FP / N)*q1.priors[0] + (FN / P)*q1.priors[1])
    gammas.append(gamma)
    gamma = gamma + gamma_inc
plt.plot(FPR, TPR, color='blue')
gamma = q1.priors[0] / q1.priors[1]
FP = np.count_nonzero(dL0 >= gamma)
N = np.size(dL0)
TP = np.count_nonzero(dL1 >= gamma)
P = np.size(dL1)
FN = np.count_nonzero(dL1 < gamma)
plt.scatter(FP / N, TP / P, color='green')
print('Min P(error): ')
print((FP / N)*q1.priors[0] + (FN / P)*q1.priors[1])
plt.title('ERM ROC Curve')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Optimal Classifier', markerfacecolor='g', markersize=10)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
L0 = D10k_val[:, [i == 0 for i in L10k_val]]
L1 = D10k_val[:, [i == 1 for i in L10k_val]]
colors0 = ['green' if dL0[i] < gamma else 'red' for i in range(len(dL0))]
colors1 = ['green' if dL1[i] >= gamma else 'red' for i in range(len(dL1))]
plt.scatter(L0[0], L0[1], marker='o', color=colors0)
plt.scatter(L1[0], L1[1], marker='^', color=colors1)
plt.title('Optimal Boundary')
plt.legend(
    handles=[Line2D([0], [0], marker='o', color='w', label='Class 0 Hit', markerfacecolor='green', markersize=10),
             Line2D([0], [0], marker='o', color='w', label='Class 0 Miss', markerfacecolor='red', markersize=10),
             Line2D([0], [0], marker='^', color='w', label='Class 1 Hit', markerfacecolor='green', markersize=10),
             Line2D([0], [0], marker='^', color='w', label='Class 1 Miss', markerfacecolor='red', markersize=10)])
plt.show()

q1.logistic_regression(D20_train, L20_train, D10k_val, L10k_val, title='20 Samples Linear')
q1.logistic_regression(D200_train, L200_train, D10k_val, L10k_val, title='200 Samples Linear')
q1.logistic_regression(D2000_train, L2000_train, D10k_val, L10k_val, title='2000 Samples Linear')

q1.logistic_regression(D20_train, L20_train, D10k_val, L10k_val, title='20 Samples Quadratic', quadratic=True)
q1.logistic_regression(D200_train, L200_train, D10k_val, L10k_val, title='200 Samples Quadratic', quadratic=True)
q1.logistic_regression(D2000_train, L2000_train, D10k_val, L10k_val, title='2000 Samples Quadratic', quadratic=True)
