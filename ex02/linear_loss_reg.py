import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
    def loss_(eps=1e-15):
        return float((y_hat - y).T.dot(y_hat - y))
    def l2():
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        return float(prime_theta.T.dot(prime_theta))
    return (loss_() + lambda_ * l2()) / (2 * y.shape[0])


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