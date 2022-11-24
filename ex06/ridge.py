import numpy as np

from mylinearregression import MyLinearRegression

class MyRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_
    
    def loss_(self, y, y_hat):
        def l2():
            prime_theta = np.array(self.thetas)
            prime_theta[0][0] = 0
            return float(prime_theta.T.dot(prime_theta))
        return super().loss_(y, y_hat) + (self.lambda_ / (2 * y.shape[0])) * l2()
    
    def loss_elem_(self, y, y_hat):
        def l2():
            prime_theta = np.array(self.thetas)
            prime_theta[0][0] = 0
            return float(prime_theta.T.dot(prime_theta))
        return super().loss_elem_(y, y_hat)  + (self.lambda_ / (2 * y.shape[0])) * l2()
    
    def gradient(self, x, y):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        prime_theta = np.array(self.thetas)
        prime_theta[0][0] = 0
        nabla_J = (x.T.dot(1 / (1 + np.e ** - x.dot(self.thetas)) - y) + self.lambda_ * prime_theta)/ x.shape[0]
        return nabla_J
    
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

