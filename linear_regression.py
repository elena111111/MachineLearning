import numpy as np
import sys
import matplotlib.pyplot as plt

'''
класс Линейная Регрессия создается с параметром self.type:
'LR' - обычная линейная регрессия
'LR_SVD' - линейная регрессия с использованием сингулярного разложения
'RR' - гребневая регрессия (ridge regression)
'RR_SVD' - гребневая регрессия с использованием сингулярного разложения
'LT' - лассо Тибширани

Параметры:
self.X - матрица объектов-признаков
self.Y - множество ответов для обучающей выборки
self.alpha - вектор коэффициентов (для результата - линейной функции)
self.V, self.D, self.U - сингулярное разложение для self.X
'''

class LinearRegression:
    def __init__(self, X, Y, typeLR):
        self.type = typeLR
        self.X = np.insert(X, 0, 1, 1)  #матрица объектов-признаков
        self.Y = Y
        print(str(self.type) + ": ")
        self.V, self.D, self.U = np.linalg.svd(self.X, full_matrices=False)  # full_matrices влияет на размерность результатов
        print("Число обусловленности: " + str(np.linalg.cond(np.diag(self.D))))  # если число обусловленности больше 10^2...10^4, матрица плохо обусл
        if np.linalg.matrix_rank(self.X) != min(self.X.shape[0], self.X.shape[1]):
            print("Матрица неполного ранга: " + str(np.linalg.matrix_rank(self.X)))
        if self.type == 'RR' or self.type == 'RR_SVD':
            self._find_tau()
        self._find_alpha()
        print("alpha:")
        print(self.alpha)


    def _find_alpha(self):
        self.V, self.D, self.U = np.linalg.svd(self.X, full_matrices=False)  # full_matrices влияет на размерность результатов

        if self.type == 'LR':
            self.alpha = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.Y)
        elif self.type == 'LR_SVD':
            F_pseudo_inverse = np.dot(np.dot(self.U, np.linalg.inv(np.diag(self.D))), np.transpose(self.V))
            self.alpha = np.dot(F_pseudo_inverse, self.Y)
        elif self.type == 'RR':
            #self.tau = 0.1
            I = np.eye(self.X.shape[1])
            A = np.linalg.inv(np.dot(np.transpose(self.X), self.X) + np.dot(self.tau, I))
            self.alpha =  np.dot(np.dot(A, np.transpose(self.X)), self.Y)
        elif self.type == 'RR_SVD':
            #self.tau = 0.1
            D1 = np.diag(self.D)
            I = np.eye(self.X.shape[1])
            A = np.linalg.inv(np.dot(D1, D1) + np.dot(self.tau, I))
            B = np.dot(np.dot(np.dot(self.U, A), np.diag(self.D)), np.transpose(self.V))
            self.alpha = np.dot(B, self.Y)


    def _find_tau(self):
        min_loo = sys.maxsize
        best_tau = 0.1
        draw_tau, draw_loo = np.arange(0.0001, 0.9999, 0.005), [] #для графика
        for t in draw_tau:
            self.tau = t
            self.alpha = self._find_alpha()
            cur_loo = self._loo()
            #print("cur_loo: " + str(cur_loo))
            draw_loo.append(cur_loo)
            if cur_loo < min_loo:
                min_loo = cur_loo
                best_tau = self.tau
        plt.ioff()
        plt.figure('loo for ' + self.type)
        plt.title('Зависимость LOO от параметра регуляризации, %s' % self.type)
        plt.plot(draw_tau, draw_loo, linewidth=1, color='blue')
        plt.plot(best_tau, min_loo, marker='o', markersize=5, color='black')
        print("min loo: " + str(min_loo))
        print("min tau: " + str(best_tau))
        self.tau = best_tau


    def _loo(self):
        res = 0
        X, Y = self.X, self.Y
        for i in range(X.shape[0]):
            self.X, self.Y = np.delete(X, [i], 0), np.delete(Y, [i])
            self._find_alpha()
            res += (self.predict(X[i]) - Y[i]) ** 2
            self.X, self.Y = X, Y
        return res


    def predict(self, test_point):
        return np.dot(self.alpha, np.transpose(test_point))


    def sse(self):  # функционал качества
        SSE = 0
        for i in range(self.X.shape[0]):
            SSE += (self.predict(self.X[i]) - self.Y[i]) ** 2
        return SSE