import numpy as np

def reg_log_loss_(y, y_hat, theta, lambda_):
    def loss_(eps=1e-15):
        v_ones = np.ones((y.shape[0], 1))
        return - float(y.T.dot(np.log(y_hat + eps)) + (v_ones - y).T.dot(np.log(v_ones - y_hat + eps))) / y.shape[0]
    def l2():
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        return float(prime_theta.T.dot(prime_theta))
    return (loss_() + (lambda_ / (2 * y.shape[0])) * l2())


def main_test():
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    print(reg_log_loss_(y, y_hat, theta, .5))
    print(reg_log_loss_(y, y_hat, theta, .05))
    print(reg_log_loss_(y, y_hat, theta, .9))




if __name__ == "__main__":
    main_test()