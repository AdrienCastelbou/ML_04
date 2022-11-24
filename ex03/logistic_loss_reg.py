import numpy as np

def reg_log_loss_(y, y_hat, theta, lambda_):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or type(theta) != np.ndarray or type(lambda_) != float:
            return None
        if y.shape[1] != 1 or y.shape != y_hat.shape or theta.shape[1] != 1:
            return None
        if len(y) == 0 or len(y_hat) == 0 or len(theta) == 0:
            return None
        v_ones = np.ones((y.shape[0], 1))
        loss = - float(y.T.dot(np.log(y_hat + 1e-15)) + (v_ones - y).T.dot(np.log(v_ones - y_hat + 1e-15))) / y.shape[0]
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        l2 = float(prime_theta.T.dot(prime_theta))
        return (loss + (lambda_ / (2 * y.shape[0])) * l2)
    except:
        return None


def main_test():
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print(reg_log_loss_(y, y_hat, theta, .5))
    print(reg_log_loss_(y, y_hat, theta, .05))
    print(reg_log_loss_(y, y_hat, theta, .9))




if __name__ == "__main__":
    main_test()