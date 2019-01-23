import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

typeOfGraphics = '3d'
typeOfData = 'Standart'

# точки для обучения
X = datasets.load_diabetes().data
Y = datasets.load_diabetes().target

if typeOfData == 'Standart':
    for j in range(X.shape[1]):
        X[:, j] = X[:, j] / np.linalg.norm(X[:, j])
        X[:, j] = X[:, j] - np.mean(X[:, j])


#точки для проверки
task = X

regressions = ['LR', 'LR_SVD', 'RR', 'RR_SVD']

if typeOfGraphics == 'none':
        #признаков много, поэтому не нужно ничего визуализировать, нужно только вывести значения коэффициентов
        for r in regressions:
            lr = LinearRegression(X, Y, r)
            print("sse: " + str(lr.sse()))

if typeOfGraphics == '2d':
    #график на плоскости по одному j - му признаку
    j = 6
    X = X[:, j:(j + 1)]
    task = task[:, j:(j + 1)]
    step = 0.01
    task = np.arange(task.min(), task.max() + step, step)
    task1 = np.column_stack((np.ones(task.shape[0]), task))
    for r in regressions:
        plt.ioff()
        plt.figure(r)
        ax = plt.subplot()
        ax.title.set_text('Выборка диабетиков. %s по %i признаку' % (r, j))
        ax.plot(X, Y, 'r.', markersize=2, color='black')
        lr = LinearRegression(X, Y, r)
        res = []
        for t in task1:
            res.append(lr.predict(t))
        ax.plot(task, res, marker='o', markersize=0, linewidth=0.7, color='red')
        print("sse: " + str(lr.sse()))
    plt.show()

if typeOfGraphics == '3d':
    # график на плоскости по двум признакам с номерами j1, j2
    #j1, j2 = 3, 6
    #j1, j2 = 3, 6
    j1, j2 = 4, 6
    X1, X2 = X[:, j1], X[:, j2]
    task1, task2 = task[:, j1], task[:, j2]
    X = np.column_stack((X1, X2))
    task = np.column_stack((task1, task2))

    step = 0.005
    task_xx, task_yy = np.arange(task[:, 0].min(), task[:, 0].max(), step), np.arange(task[:, 1].min(), task[:, 1].max(), step)
    xx, yy = np.meshgrid(task_xx, task_yy)
    for r in regressions:
        fig = plt.figure(r)
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text('Выборка Бостон. %s по %i и %i признакам' % (r, j1, j2))
        ax.scatter(X[:, 0], X[:, 1], Y, color='black', label='Обучающая выборка')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.63))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        lr = LinearRegression(X, Y, r)
        res = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                test = [1., xx[i, j], yy[i, j]]
                res.append(lr.predict(test))
        res = np.asarray(res)
        res.shape = (xx.shape[0], xx.shape[1])

        surf = ax.plot_surface(xx, yy, res, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.4)
        fig.colorbar(surf, shrink=0.5, aspect=5, orientation='horizontal')
        print("sse: " + str(lr.sse()))
    plt.show()