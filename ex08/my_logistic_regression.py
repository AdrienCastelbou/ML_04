import numpy as np

class MyLogisticRegression:

    supported_penalties = ["l2"]
    
    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty="l2", lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta.astype(float)
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0


    def gradient(self, x, y):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray or type(self.theta) != np.ndarray:
                return None
            l = len(x)
            y_hat = self.predict_(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            nabla_J = x.T.dot(y_hat - y) / l
            return nabla_J
        except:
            return None

    def fit_(self, x, y):
        try:
            if type(x) != np.ndarray or type(y) != np.ndarray or type(self.theta) != np.ndarray or type(self.alpha) != float or type(self.max_iter) != int:
                return None
            for i in range(self.max_iter):
                nabla_J = self.gradient(x, y)
                self.theta -= self.alpha * nabla_J
            return self.theta
        except:
            return None
    
    def predict_(self, x):
        try:
            if type(x) != np.ndarray or type(self.theta) != np.ndarray:
                return None
            if not len(x) or not len(self.theta):
                return None
            extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
            return 1 / (1 + np.e ** - extended_x.dot(self.theta))
        except:
            return None

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
            return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        except:
            return None
    
    def loss_(self, y, y_hat):
        try:
            if type(y) != np.ndarray or type(y_hat) != np.ndarray:
                return None
            if y.shape[1] != 1 or y.shape != y_hat.shape:
                return None
            y_hat = y_hat
            l = len(y)
            v_ones = np.ones((l, 1))
            return - float(y.T.dot(np.log(y_hat + 1e-15)) + (v_ones - y).T.dot(np.log(1 - y_hat + 1e-15))) / l
        except:
            return None
