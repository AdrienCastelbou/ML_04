import numpy as np

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas.astype(float)


    def gradient(self, x, y):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray or type(self.thetas) != np.ndarray:
                return None
            l = len(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            nabla_J = x.T.dot(x.dot(self.thetas) - y) / l
            return nabla_J
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
    
    def predict_(self, x):
            if type(x) != np.ndarray or type(self.thetas) != np.ndarray:
                return None
            if not len(x) or not len(self.thetas):
                return None
            extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
            return extended_x.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        try:
            if type(y) != np.ndarray or type(y_hat) != np.ndarray:
                return None
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                return None
            return (y_hat - y) ** 2
        except:
            return None
    
    def loss_(self, y, y_hat):
        try:
            if type(y) != np.ndarray or type(y_hat) != np.ndarray:
                return None
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                return None
            return float(1 / (2 * y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
        except:
            return None

    @staticmethod
    def mse_(y, y_hat):
        if type(y) != np.ndarray or type(y_hat) != np.ndarray:
            return None
        return float(1 / (y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
