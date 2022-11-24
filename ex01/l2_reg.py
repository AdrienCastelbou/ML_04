import numpy as np

def iterative_l2(theta):
    try:
        if type(theta) != np.ndarray or theta.shape[1] != 1 or len(theta) == 0:
            return None
        res = 0
        for i in range(theta.shape[0]):
            if i > 0:
                res += theta[i] ** 2
        return float(res)
    except:
        return None

def l2(theta):
    try:
        if type(theta) != np.ndarray or theta.shape[1] != 1 or len(theta) == 0:
            return None
        prime_theta = np.array(theta)
        prime_theta[0][0] = 0
        return float(prime_theta.T.dot(prime_theta))
    except:
        return None


def main_test():
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(iterative_l2(x))
    print(l2(x))
    y = np.array([3,0.5,-6]).reshape((-1, 1))
    print(iterative_l2(y))
    print(l2(y))

if __name__ == "__main__":
    main_test()