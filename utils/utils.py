import numpy as np

def data_cross_splitter(x, y, k=10):
    x_cross = np.array_split(x, k)
    y_cross = np.array_split(y, k)
    return x_cross, y_cross

def normalize(x):
    norm_x = np.array([])
    for col in x.T:
        mean_col = np.mean(col)
        std_col = np.std(col)
        n_col = ((col - mean_col) / std_col).reshape((-1, 1))
        if norm_x.shape == (0,):
            norm_x = n_col
        else:
            norm_x = np.hstack((norm_x, n_col))
    return norm_x

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


def accuracy_score_(y, y_hat):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape:
            return None
        t = 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_i == y_hat_i:
                t += 1
        return t / len(y)
    except:
        return None

def precision_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        tp, fp = 0, 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_hat_i == pos_label and y_i == y_hat_i:
                tp += 1
            elif y_hat_i == pos_label and y_i != y_hat_i:
                fp += 1
        return tp / (tp + fp)
    except:
        return None

def recall_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        tp, fn = 0, 0
        for y_i, y_hat_i in zip(y, y_hat):
            if y_hat_i == pos_label and y_i == y_hat_i:
                tp += 1
            elif y_i == pos_label and y_i != y_hat_i:
                fn += 1
        return tp / (tp + fn)
    except:
        return None

def f1_score_(y, y_hat, pos_label=1):
    try:
        if type(y) != np.ndarray or type(y_hat) != np.ndarray or y.shape != y_hat.shape or not isinstance(pos_label, (str, int)):
            return None
        prec = precision_score_(y, y_hat, pos_label)
        recall = recall_score_(y, y_hat, pos_label)
        return (2 * prec * recall) / (prec + recall)
    except:
        return None

def multiclass_f1_score_(y, y_hat, labels):
    try:
        f1_score_mean = 0
        for label in labels:
            f1_score_mean += f1_score_(y, y_hat, label)
        return f1_score_ / len(labels)
    except:
        return None