import numpy as np
import math
import matplotlib.pyplot as plt

class NonparamRegression:
    def __init__(self, X, Y, ker):
        self.ker = ker
        self.X = X
        self.Y = Y
        self.h = self.__find_h_opt(0.01, round((X.max() - X.min()) / 4), round((X.max() - X.min()) / X.shape[0], 2))

    def __e_dist(self, u, v):
        return math.sqrt((u - v) ** 2)

    def get_h(self):
        return self.h

    def set_h(self, h):
        self.h = h

    def __find_h_opt(self, min_h, max_h, step_h):
        loo_min = 50000
        plt.figure()
        for h in np.arange(min_h, max_h, step_h):
            cur_loo = self.__loo(h)
            plt.plot(h, cur_loo, linewidth=2, color='orange')
            plt.xlabel('h')
            plt.ylabel('loo')
            print("loo: " + str(cur_loo))
            print("h: " + str(h))
            if cur_loo < loo_min:
                loo_min = cur_loo
                h_opt = h
        print("h_opt: " + str(h_opt))
        print("loo_min: " + str(loo_min))
        plt.title("Подбор ширины окна для непараметрической регрессии (оптимальное h = %f)" % h_opt)
        plt.show()
        return h_opt

    def __loo(self, h):
        res = 0
        for i in range(self.X.shape[0]):
            res += (self.__predict_for_loo(self.X[i], np.delete(self.X, [i]), np.delete(self.Y, [i]), h, self.ker) - self.Y[i]) ** 2
        return res

    def __predict_for_loo(self, test_point, X, Y, h, ker):  #классифицирует одну точку
        numerator, denominator = 0, 0
        for i in range(X.shape[0]):
            numerator += Y[i] * ker(self.__e_dist(test_point, X[i]) / h)
            denominator += ker(self.__e_dist(test_point, X[i]) / h)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator  # alpha

    def predict(self, test_point):  #классифицирует одну точку
        numerator, denominator = 0, 0
        for i in range(self.X.shape[0]):
            numerator += self.Y[i] * self.ker(self.__e_dist(test_point, self.X[i]) / self.h)
            denominator += self.ker(self.__e_dist(test_point, self.X[i]) / self.h)
        if denominator == 0:
            return 0
        else:
            return numerator / denominator  # alpha