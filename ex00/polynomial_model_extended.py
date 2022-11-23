import numpy as np

def add_polynomial_features(x, power):
    poly_x = np.array(x)
    for i in range(2, power + 1):
        for col in x.T:
            poly_x = np.hstack((poly_x, (col ** i).reshape((-1, 1))))
    return poly_x


def main_test():
    x = np.arange(1,11).reshape(5, 2)
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 4))


if __name__ == "__main__":
    main_test()