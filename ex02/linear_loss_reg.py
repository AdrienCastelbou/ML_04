import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or type(theta) != np.ndarray or type(lambda_) != float:
            return None
        if y.shape[1] != 1 or y.shape != y_hat.shape or theta.shape[1] != 1:
            return None
        if len(y) == 0 or len(y_hat) == 0 or len(theta) == 0:
            return None
        loss_ = float((y_hat - y).T.dot(y_hat - y))
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        l2 = float(prime_theta.T.dot(prime_theta))
        return (loss_ + lambda_ * l2) / (2 * y.shape[0])
    except:
        return None


def main_test():
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    # Example :
    print(reg_loss_(y, y_hat, theta, .5))
    print(reg_loss_(y, y_hat, theta, .05))
    print(reg_loss_(y, y_hat, theta, .9))


if __name__ == "__main__":
    main_test()