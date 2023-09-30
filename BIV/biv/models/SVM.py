import numpy as np
import scipy.optimize
from utils import vrow, vcol
from .IModel import IModel


class SupportVectorMachine(IModel):
    def __init__(self, C, K):
        self.C = C
        self.K = K
        # print(f'K = {K}, C = {C}')

    def _Ld(self, alpha):
        Ha = np.dot(self.__H, alpha)
        f = 0.5*np.dot(alpha.T, Ha)-alpha.sum()
        grad = Ha-1
        return f, grad

    # def _J(self, wcap):
    #     loss = 1-self.z*np.dot(wcap.T, self.Dext)
    #     loss = np.clip(loss, a_min=0, a_max=None)
    #     return 0.5*np.dot(wcap.T, wcap).sum()+self.C*loss.sum()

    def train(self, DTR, LTR):
        # print('training started')
        Dext = np.vstack([DTR, self.K*np.ones(DTR.shape[1])])
        z = vrow(LTR*2-1)

        x0 = np.zeros(DTR.shape[1])
        bounds = [(0, self.C)]*DTR.shape[1]
        self.__H = np.dot(Dext.T, Dext) * np.dot(z.T, z)
        alpha, fval, _ = scipy.optimize.fmin_l_bfgs_b(
            self._Ld, x0, bounds=bounds, factr=1.0)
        alpha = vcol(alpha)

        wcap = np.dot(alpha.T, (z*Dext).T)
        wcap = wcap.reshape(Dext.shape[0], 1)
        self.w, self.b = wcap[0:-1], wcap[-1]*self.K
        # print('training ended')
        # p = self._J(wcap)

        # print(f'Primal loss\t=\t{p}')
        # print(f'Dual loss\t=\t{-fval}')
        # print(f'Duality gap\t=\t{p+fval}')

    def test(self, DTE):
        if self.w is None or self.b is None:
            raise AttributeError()
        s = np.dot(self.w.T, DTE)+self.b
        return s > 0, s

    @staticmethod
    def load(file_name):
        raise NotImplementedError()


class Kernel:
    @staticmethod
    def dot():
        def kernel_function(x1, x2):
            return np.dot(x1.T, x2)
        return kernel_function

    @staticmethod
    def poly(d, c):
        # print(f'd = {d}, c = {c}, ', end='')
        def kernel_function(x1, x2):
            return (np.dot(x1.T, x2)+c)**d
        return kernel_function

    @staticmethod
    def RBF(gamma):
        # print(f'g = {gamma}, ', end='')
        def kernel_function(x1, x2):
            # add a third dimension to exploit broadcasting
            # and compute pairwise distances
            x1 = np.expand_dims(x1, axis=2)
            x2 = np.expand_dims(x2, axis=1)
            dist = np.sum((x1-x2)**2, axis=0)

            return np.exp(-gamma*dist)
        return kernel_function


class NonLinearSupportVectorMachine(IModel):
    def __init__(self, C, K, kernel=Kernel.dot()):
        self.C = C
        self.K = K
        self.kernel = NonLinearSupportVectorMachine._adapt_kernel(kernel, K*K)
        # print(f'K = {K}, C = {C}')

    @staticmethod
    # this function is just used to correct the kernel function, adding the value of xi
    def _adapt_kernel(kernel, xi):
        def kernel_function(x1, x2):
            return kernel(x1, x2) + xi
        return kernel_function

    def _Ld(self, alpha):
        Ha = np.dot(self.__H, alpha)
        f = 0.5*np.dot(alpha.T, Ha)-vrow(alpha).sum()
        grad = Ha-1
        return f, grad

    def train(self, DTR, LTR):
        # print('training started')
        self.DTR = DTR
        self.z = vrow(LTR*2-1)
        self.__H = np.dot(self.z.T, self.z) * self.kernel(DTR, DTR)

        x0 = np.zeros(DTR.shape[1])
        bounds = [(0, self.C)]*DTR.shape[1]
        alpha, fval, _ = scipy.optimize.fmin_l_bfgs_b(self._Ld, x0, bounds=bounds, maxiter=1000, factr=1.0)
        self.alpha = vrow(alpha)
        # print(f'Dual loss\t=\t{-fval}')
        # print('training ended')

    def test(self, DTE):
        if self.alpha is None:
            raise AttributeError()
        k = self.kernel(self.DTR, DTE)
        za = self.z*self.alpha
        s = np.dot(za, k).sum(0)
        return s > 0, s

    @staticmethod
    def load(file_name):
        raise NotImplementedError()
