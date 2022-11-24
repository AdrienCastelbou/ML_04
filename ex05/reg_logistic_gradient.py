import numpy as np
import math

def logistic_predict_(x, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if not len(x) or not len(theta):
            return None
        extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
        return 1 / (1 + math.e ** - extended_x.dot(theta))
    except:
        return None


def reg_logistic_grad(y, x, theta, lambda_):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray or not isinstance(lambda_, (int, float)):
            return None
        if theta.shape[0] != x.shape[1] + 1:
            return None
        if y.shape[1] != 1:
            return None
        if not len(x) or not len(y) or not len(theta):
            return None 
        reg_nabla = np.zeros(theta.shape)
        for x_i, y_i, y_hat_i in zip(x, y, logistic_predict_(x, theta)):
            reg_nabla[0] += y_hat_i - y_i
            reg_nabla[1:] += (y_hat_i - y_i) * x_i.reshape(-1, 1)
        reg_nabla[1:] += lambda_ * theta[1:]
        return reg_nabla / x.shape[0]
    except:
        return None

def vec_reg_logistic_grad(y, x, theta, lambda_):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray or not isinstance(lambda_, (int, float)):
            return None
        if theta.shape[0] != x.shape[1] + 1:
            return None
        if y.shape[1] != 1:
            return None
        if not len(x) or not len(y) or not len(theta):
            return None 
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        nabla_J = (x.T.dot(1 / (1 + np.e ** - x.dot(theta)) - y) + lambda_ * prime_theta)/ x.shape[0]
        return nabla_J 
    except:
        return None

def main_test():
    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(reg_logistic_grad(y, x, theta, 1)) 
    print(vec_reg_logistic_grad(y, x, theta, 1))
    print(reg_logistic_grad(y, x, theta, 0.5)) 
    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    print(reg_logistic_grad(y, x, theta, 0.0)) 
    print(vec_reg_logistic_grad(y, x, theta, 0.0))


if __name__ == "__main__":
    main_test()