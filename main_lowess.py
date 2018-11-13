import numpy as np
from sklearn import datasets
import math
import matplotlib.pyplot as plt
from lowess import NonparamRegression

def ker_quart(z):
    if abs(z) <= 1:
        return (1 - z*z)*(1 - z*z)
    else:
        return 0

def ker_gauss(z):
    return 1/math.sqrt(2*math.pi) * math.exp(-0.5*z*z)



j = 5   #номер признака
X = datasets.load_boston().data[:, j]
Y = datasets.load_boston().target

beg = 380
end = 410
X = X[beg:end]
Y = Y[beg:end]

step = (X.max() - X.min()) / X.shape[0]
ker = ker_quart
nr = NonparamRegression(X, Y, ker)

task = np.arange(X.min(), X.max(), step)
plt.figure()
plt.xlim(task.min(), task.max())
plt.ylim(Y.min(), Y.max() + 1)

plt.plot(X, Y, 'r.', markersize=2, color='black')
alpha = []
for t in task:
    alpha.append(nr.predict(t))
plt.plot(task, alpha, linewidth=2, color='blue')
plt.title("LOWESS (выборка \"Бостон\" по %i признаку, h = %f, кварт. ядро)" % (j, nr.get_h()))
plt.legend(('Обучающая выборка', 'Результат'), loc='lower right')
plt.show()

print("sse: " + str(nr.sse()))



