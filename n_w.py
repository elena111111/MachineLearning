import numpy as np
import math
import matplotlib.pyplot as plt

class NonparamRegression:
    def __init__(self, X, Y, ker):
        self.ker = ker
        self.X = X
        self.Y = Y
        self.h = self._find_h_opt(0.01, round((X.max() - X.min()) / 3), round((X.max() - X.min()) / X.shape[0], 2))

    def _e_dist(self, u, v):
        return math.sqrt((u - v) ** 2)

    def get_h(self):
        return self.h

    def set_h(self, h):
        self.h = h

    def _find_h_opt(self, min_h, max_h, step_h):
        loo_min = 50000
        plt.figure()
        СurLoo = []
        H = np.arange(min_h, max_h, step_h)
        for h in H:
            cur_loo = self._loo(h)
            СurLoo.append(cur_loo)
            print("loo: " + str(cur_loo))
            print("h: " + str(h))
            if cur_loo < loo_min:
                loo_min = cur_loo
                h_opt = h
        plt.plot(H, СurLoo, linewidth=2, color='orange')
        plt.xlabel('h')
        plt.ylabel('loo')
        print("h_opt: " + str(h_opt))
        print("loo_min: " + str(loo_min))
        plt.title("Подбор ширины окна для непараметрической регрессии (оптимальное h = %f)" % h_opt)
        plt.show()
        return h_opt

    def _loo(self, h):
        res = 0
        for i in range(self.X.shape[0]):
            res += (self._predict(self.X[i], np.delete(self.X, [i]), np.delete(self.Y, [i]), h, self.ker) - self.Y[i]) ** 2
        return res

    def _predict(self, test_point, X, Y, h, ker):  #классифицирует одну точку
        numerator, denominator = 0, 0
        for i in range(X.shape[0]):
            numerator += Y[i] * ker(self._e_dist(test_point, X[i]) / h)
            denominator += ker(self._e_dist(test_point, X[i]) / h)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator  # alpha

    def predict(self, test_point):  #классифицирует одну точку
        numerator, denominator = 0, 0
        return self._predict(test_point, self.X, self.Y, self.h, self.ker)

    def sse(self):  #функционал качества
        SSE = 0
        for i in range(self.X.shape[0]):
            SSE += (self.predict(self.X[i]) - self.Y[i])**2
        return SSE