import numpy as np


def arg_checker(fnc):
    def ckeck_args(*args, **kwargs):
        for arg in args:
            if type(arg) == MyLogisticRegression:
                if type(arg.alpha) != float or type(arg.max_iter) != int or not isinstance(arg.lambda_, (float, int))\
                    or not arg.penality in arg.supported_penalties or type(arg.theta) != np.ndarray:
                    print("Bad params for your model")
                    return None
            else:
                if type(arg) != np.ndarray:
                    print(f"Bad param for {fnc.__name__}")
                    return None
        return fnc(*args, **kwargs)
    return ckeck_args


class MyLogisticRegression:

    supported_penalties = ["l2"]
    
    def __init__(self, theta, alpha=0.001, max_iter=1000, penality="l2", lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta.astype(float)
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalties else 0

    def l2(self):
        try:
            prime_theta = np.array(self.theta)
            prime_theta[0][0] = 0
            l2 = float(prime_theta.T.dot(prime_theta))
            return l2
        except:
            return None

    @arg_checker
    def gradient(self, x, y):
        try:
            l = len(x)
            y_hat = self.predict_(x)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
            prime_theta = np.array(self.theta)
            prime_theta[0][0] = 0
            nabla_J = (x.T.dot(y_hat - y) + self.lambda_ * prime_theta )/ l
            return nabla_J
        except:
            return None

    @arg_checker
    def fit_(self, x, y):
        try:
            for i in range(self.max_iter):
                nabla_J = self.gradient(x, y)
                self.theta -= self.alpha * nabla_J
            return self.theta
        except:
            return None
    
    @arg_checker
    def predict_(self, x):
        try:
            if not len(x) or not len(self.theta):
                return None
            extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
            return 1 / (1 + np.e ** - extended_x.dot(self.theta))
        except:
            return None

    @arg_checker
    def loss_elem_(self, y, y_hat):
        try:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], -1)
            if y_hat.ndim == 1:
                y_hat = y_hat.reshape(y_hat.shape[0], -1)
            if y.shape[1] != 1 or y_hat.shape[1] != 1:
                return None
            return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        except:
            return None
    
    @arg_checker
    def loss_(self, y, y_hat):
        try:
            if y.shape[1] != 1 or y.shape != y_hat.shape:
                return None
            y_hat = y_hat
            l = len(y)
            v_ones = np.ones((l, 1))
            loss = - float(y.T.dot(np.log(y_hat + 1e-15)) + (v_ones - y).T.dot(np.log(1 - y_hat + 1e-15))) / l
            return loss + (self.lambda_ / (2 * y.shape[0])) * self.l2()
        except:
            return None
