import numpy as np

def predict_(x, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if not len(x) or not len(theta):
            return None
        extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
        return extended_x.dot(theta)
    except:
        return None

def reg_linear_grad(y, x, theta, lambda_):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray or not isinstance(lambda_, (int, float)):
            return None
        if theta.shape[0] != x.shape[1] + 1:
            return None
        if y.shape[1] != 1:
            return None
        if not len(x) or not len(y) or not len(theta):
            return None
        nabla_J = np.zeros(theta.shape)
        for x_i, y_i, y_hat_i in zip(x, y, predict_(x, theta)):
            nabla_J[0] += (y_hat_i - y_i)
            nabla_J[1:] += (y_hat_i - y_i) * x_i.reshape(-1, 1)
        nabla_J[1: ] += lambda_ * theta[1:]
        return nabla_J / y.shape[0]
    except:
        return None


def vec_reg_linear_grad(y, x, theta, lambda_):
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
        nabla_J = (x.T.dot(x.dot(theta) - y) + lambda_ * prime_theta)/ x.shape[0]
        return nabla_J
    except:
        return None

def main_test():
    x = np.array([[ -6, -7, -9], [ 13, -2, 14], [ -7, 14, -1], [-8, -4, 6], [-5, -9, 6], [ 1, -5, 11], [9,-11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    print(reg_linear_grad(y, x, theta, 1))
    print(vec_reg_linear_grad(y, x, theta, 1))
    print(reg_linear_grad(y, x, theta, 0.5))
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    print(reg_linear_grad(y, x, theta, 0.0))
    print(vec_reg_linear_grad(y, x, theta, 0.0))

if __name__ == "__main__":
    main_test()