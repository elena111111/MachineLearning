import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import median

class NonparamRegression:
    def __init__(self, X, Y, ker):
        self.ker = ker
        self.X = X
        self.Y = Y
        self.deviation = 1000  # погрешность для подбора gamma, зависит от h
        self.med_eps = 1  # для gamma
        self.gamma = [1]*self.X.shape[0]
        self.h = 0.55
        self._find_h_and_gamma(0.01, round((X.max() - X.min()) / 1), round((X.max() - X.min()) / X.shape[0], 2), self._ker_quart_modif)


    def _e_dist(self, u, v):
        return math.sqrt((u - v) ** 2)

    def get_h(self):
        return self.h

    def set_h(self, h):
        self.h = h

    def _ker_quart_modif(self, x):
        x /= (6 * self.med_eps)
        if abs(x) <= 1:
            return (1 - x * x) * (1 - x * x)
        else:
            return 0


    def _loop_for_gamma(self, gamma, h, ker):
        alpha = []
        eps = []
        print("gamma1: " + str(gamma))
        for i in range(self.X.shape[0]):
            alpha.append(self._predict(self.X[i], np.delete(self.X, [i]), np.delete(self.Y, [i]), np.delete(gamma, [i]), h, ker))
            gamma[i] = ker(abs(alpha[i] - self.Y[i]))
            eps.append(alpha[i] - self.Y[i])
        self.med_eps = median(eps)
        print("gamma2: " + str(gamma))
        return

    def _find_h_and_gamma(self, min_h, max_h, step_h, ker_gammas):
        loo_min = 1000000
        plt.figure()
        CurLoo = []
        H = np.arange(min_h, max_h, step_h)

        for h in H:
            print("h: " + str(h))
            gamma = [1] * self.X.shape[0]
            cur_gamma = list(gamma) #иначе будет передача по ссылке
            self._loop_for_gamma(gamma, h, ker_gammas)
            while(abs(self._loo(gamma, h) - self._loo(cur_gamma, h)) >= self.deviation):
                cur_gamma = list(gamma)
                self._loop_for_gamma(gamma, h, ker_gammas)
            cur_loo = self._loo(gamma, h)
            CurLoo.append(cur_loo)
            if cur_loo < loo_min:
                loo_min = cur_loo
                self.gamma = gamma
                self.h = h
        plt.plot(H, CurLoo, linewidth=2, color='orange')
        plt.xlabel('h')
        plt.ylabel('loo')
        print("h_opt: " + str(self.h))
        print("loo_min: " + str(loo_min))
        print("gamma: " + str(gamma))
        plt.title("Подбор ширины окна для LOWESS (оптимальное h = %f)" % self.h)
        plt.show()
        return

    def _loo(self, gamma, h):
        res = 0
        for i in range(self.X.shape[0]):
            res += (self._predict(self.X[i], np.delete(self.X, [i]), np.delete(self.Y, [i]), np.delete(gamma, [i]), h, self.ker) - self.Y[i]) ** 2
        print("loo: " + str(res))
        return res

    def _predict(self, test_point, X, Y, gamma, h, ker):  #классифицирует одну точку
        numerator, denominator = 0, 0
        for i in range(X.shape[0]):
            numerator += Y[i] * gamma[i] * ker(self._e_dist(test_point, X[i]) / h)
            denominator += gamma[i] * ker(self._e_dist(test_point, X[i]) / h)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator  # alpha

    def predict(self, test_point):  #классифицирует одну точку
        return self._predict(test_point, self.X, self.Y, self.gamma, self.h, self.ker)

    def sse(self):  #функционал качества
        SSE = 0
        for i in range(self.X.shape[0]):
            SSE += (self.predict(self.X[i]) - self.Y[i])**2
        return SSE

