import numpy as np
import sys
sys.path.append('../')
from utils.mylinearregression import MyLinearRegression

class MyRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_

    def get_params_(self):
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'thetas': self.thetas,
            'lambda_': self.lambda_
        }
    
    def set_params_(self, params):
        try:
            for p in params:
                if p == 'alpha':
                    if type(params[p]) != float:
                        raise Exception
                    self.alpha = params[p]
                elif p == 'max_iter':
                    if type(params[p]) != int:
                        raise Exception
                    self.max_iter = params[p]
                elif p == 'thetas':
                    if type(params[p]) != np.ndarray:
                        raise Exception
                    self.thetas = params[p]
                elif p == 'lambda_':
                    if type(params[p]) != float:
                        raise Exception
                    self.lambda_ = params[p]
            return self
        except:
            return None

    def loss_(self, y, y_hat):
        try:
            if type(y) != np.ndarray or type(y_hat) != np.ndarray:
                return None
            if y.shape[1] != 1 or y.shape != y_hat.shape:
                return None
            if len(y) == 0 or len(y_hat) == 0:
                return None
            loss_ = float((y_hat - y).T.dot(y_hat - y))
            prime_theta = np.array(self.thetas)
            prime_theta[0][0] = 0
            l2 = float(prime_theta.T.dot(prime_theta))
            return (loss_ + self.lambda_ * l2) / (2 * y.shape[0])
        except:
            return None
        
    def loss_elem_(self, y, y_hat):
        try:
            return super().loss_elem_(y, y_hat)
        except:
            return None

    def predict_(self, x):
        try:
            return super().predict_(x)
        except:
            return None

    def gradient_(self, x, y):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray:
                return None
            if self.thetas.shape[0] != x.shape[1] + 1:
                return None
            if y.shape[1] != 1:
                return None
            if not len(x) or not len(y):
                return None 
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            prime_theta = np.array(self.thetas)
            prime_theta[0][0] = 0
            reg_gradient = (x.T.dot(x.dot(self.thetas) - y) + self.lambda_ * prime_theta)/ x.shape[0]
            return reg_gradient
        except:
           return None
    
    def fit_(self, x, y):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray or type(self.thetas) != np.ndarray or type(self.alpha) != float or type(self.max_iter) != int:
                return None
            for i in range(self.max_iter):
                nabla_J = self.gradient(x, y)
                self.thetas -= self.alpha * nabla_J
            return self.thetas
        except:
            return None