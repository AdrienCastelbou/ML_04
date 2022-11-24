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
    
    def set_params(self, params):
        for p in params:
            if p == 'alpha':
                self.alpha = params[p]
            elif p == 'max_iter':
                self.max_iter = params[p]
            elif p == 'thetas':
                self.thetas = params[p]
            elif p == 'lambda_':
                self.lambda_ = params[p]
        return self

