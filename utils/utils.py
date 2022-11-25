import numpy as np

def data_spliter(x, y, proportion):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(proportion) != float:
            return None
        n_train = int(proportion * x.shape[0])
        n_test = int((1-proportion) * x.shape[0]) + 1
        perm = np.random.permutation(len(x))
        s_x = x[perm]
        s_y = y[perm]
        x_train, y_train = s_x[:n_train], s_y[:n_train]
        x_test, y_test =  s_x[-n_test:], s_y[-n_test:]
        return x_train, x_test, y_train, y_test
    except:
        return None


def zscore(x):
    try:
        if type(x) != np.ndarray or len(x) == 0:
            return None
        if x.ndim == 1:    
            x = x.reshape(-1, 1)
        if x.shape[0] != 1 and x.shape[1] != 1:
            return None
        normalized = np.zeros(x.shape)
        mean_ = np.mean(x)
        std_ = np.std(x)
        for i in range(len(x)):
            normalized[i] = (x[i] - mean_) / std_
        return normalized
    except:
        return None

def add_polynomial_features(x, power):
    try:
        if type(x) != np.ndarray or not isinstance(power, (int, float)):
            return None
        poly_x = np.array(x)
        for i in range(2, power + 1):
            for col in x.T:
                poly_x = np.hstack((poly_x, (col ** i).reshape((-1, 1))))
        return poly_x
    except:
        return None