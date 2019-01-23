import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, X):
        self.X = X
        v, w = la.eig(np.dot(np.transpose(self.X), self.X)) #собств знач и векторы
        print("v: " + str(v))
        #print("w: " + str(w))
        self.Eigens = np.column_stack((np.asarray(v), np.asarray(w)))
        self.Eigens = np.asarray(sorted(self.Eigens, key=lambda x: x[0], reverse=True))

        #if m <= np.linalg.matrix_rank(self.X):
        print("rank(X): " + str(np.linalg.matrix_rank(self.X)))
        self._find_m()
        self.U = self.Eigens[:, 1:(self.m + 1)]
        self.G = np.dot(self.X, self.U)

        #print("U: " + str(self.U))
        #print("G: " + str(self.G))


    def _find_m(self):
        eig_val = self.Eigens[:, 0]
        eps = np.arange(0., 0.9999, 0.005)
        denom = sum(eig_val)
        #best_m = np.linalg.matrix_rank(self.X)
        best_Em, best_Eps, best_draw_E, best_draw_m = 10, 10, [],[]
        for e in eps:
            m, Em = 1, e + 1.
            draw_m, draw_E = [], []
            Em1 = Em
            while Em > e:
                numer = sum(eig_val[(m + 1):])
                Em = numer / denom
                draw_E.append(Em)
                draw_m.append(m)
                if Em < best_Em:
                    best_Eps, best_Em = e, Em
                    best_draw_E, best_draw_m = draw_E, draw_m
                m += 1
                Em1 = Em
        self.m = 3 #посмотрели излом по графику
        plt.ioff()
        plt.figure('pca')
        plt.xlabel('m')
        plt.ylabel('E(m)')
        plt.title('Относительная погрешность E(m) и число компонент m, при eps = %i' % float(best_Eps))
        plt.plot(best_draw_m, best_draw_E, marker='o', markersize=5, color='blue')
        plt.show()


'''def _find_U(self):
        #print("w: " + str(w))
        #print("v: " + str(v))
        Eigens = np.column_stack((np.asarray(w), np.asarray(v)))
        #print("Eigens before: " + str(Eigens))
        sorted(Eigens, key=lambda x: x[0], reverse=True)
        #print("Eigens after: " + str(Eigens))
        #print("shape: " + str(Eigens.shape))
        self.U = Eigens[:, 1:(self.m + 1)]'''