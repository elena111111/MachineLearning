import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

def e_dist(u, v):
  return math.sqrt((u - v)**2)

def ker_rect(z):
    if abs(z) <= 1:
        return 1
    else:
        return 0

def ker_gauss(z):
    return 1/math.sqrt(2*math.pi) * math.exp(-0.5*z*z)

def loo(func, X, Y, h, ker):
    res = 0
    for i in range(X.shape[0]):
        res += (nad_wat(X[i], np.delete(X, [i]), np.delete(Y, [i]), h, ker) - Y[i])**2
    return res

def nad_wat(test_point, X, Y, h, ker):
    numerator, denominator = 0, 0
    for i in range(X.shape[0]):
        numerator += Y[i]*ker(e_dist(test_point, X[i]) / h)
        denominator += ker(e_dist(test_point, X[i]) / h)
    if denominator == 0:
        return 0
    else:
        return numerator / denominator  #alpha

def get_h_opt(min_h, max_h, step_h):
    loo_min = 50000
    plt.figure()
    for h in np.arange(min_h, max_h, step_h):
        cur_loo = loo(nad_wat, X, Y, h, ker)
        plt.plot(h, cur_loo, 'r.', markersize=2, color='orange')
        plt.xlabel('h')
        plt.ylabel('loo')
        print("loo: " + str(cur_loo))
        print("h: " + str(h))
        if cur_loo < loo_min:
            loo_min = cur_loo
            h_opt = h
    print("h_opt: " + str(h_opt))
    plt.title("Подбор ширины окна для непараметрической регрессии (оптимальное h = %f)" % h_opt)
    plt.show()
    return h_opt

j = 5   #номер признака
X = datasets.load_boston().data[:, j]
Y = datasets.load_boston().target

step = (X.max() - X.min()) / X.shape[0]
#step = 3
ker = ker_gauss
#h = 0.1 #ширина окна

h_opt = get_h_opt(0.01, round((X.max() - X.min()) / 2), round((X.max() - X.min()) / X.shape[0], 2))
#h_opt = 0.35

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

task = np.arange(X.min(), X.max(), step)
plt.figure()
plt.xlim(task.min(), task.max() + 1)
plt.ylim(Y.min(), Y.max() + 10)

plt.plot(X, Y, 'r.', markersize=2, color='black')
for t in task:
    alpha = nad_wat(t, X, Y, h_opt, ker_rect)
    plt.plot(t, alpha, 'r.', markersize=2, color='blue')
plt.plot(X, Y, 'r.', markersize=2, color='black')
plt.title("Непараметрическая регрессия (выборка \"Бостон\" по %i признаку, h = %f, гаусс. ядро)" % (j, h_opt))
plt.legend(('Обучающая выборка', 'Результат'), loc='upper left')
plt.show()
