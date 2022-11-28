import sys
sys.path.append('../')
from utils.utils import *
from ex06.ridge import MyRidge
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def save_models(results):
    file = open('zscore-models.pickle', 'wb')
    pickle.dump(results, file)
    file.close()

def perform_regression(x, y, lambda_):
    print("start")
    theta = np.random.rand(x.shape[1] + 1, 1).reshape(-1, 1)
    myR =  MyRidge(theta, alpha = 1e-2, max_iter = 1000000, lambda_=lambda_)
    myR.fit_(x, y)
    print("end")
    return myR


def regression_engine(x, y):
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
    x_cross, y_cross = data_cross_splitter(x, y)
    x_train = add_polynomial_features(x_train, 4)
    x_train = normalize(x_train)
    models = {}
    for w_rank in range(1, 5):
        w_features = [0, 3, 6, 9][:w_rank]
        for d_rank in range(1, 5):
            d_features = [1, 4, 7, 10][:d_rank]
            for t_rank in range(1, 5):
                t_features = [2, 5, 8, 11][:t_rank]
                x_train_features = x_train[:, np.concatenate((w_features, d_features, t_features))]
                for l, lambda_ in enumerate([0., 0.2, 0.4, 0.6, 0.8]):
                    print(f"w{w_rank}d{d_rank}t{t_rank}λ{lambda_}")
                    current = perform_regression(x_train_features, y_train, lambda_)
                    models[f"w{w_rank}d{d_rank}t{t_rank}λ{lambda_}"] = current#perform_regression(x_train_features, y_train, lambda_)
    save_models(models)


def load_datasets():
    content = pd.read_csv("space_avocado.csv")
    X = np.array(content[["weight", "prod_distance", "time_delivery"]])
    if X.shape[1] !=  3:
        raise Exception("Datas are missing in space_avocado.csv")        
    Y = np.array(content[["target"]])
    if Y.shape[1] !=  1:
        raise Exception("Datas are missing in space_avocado.csv")   
    return X, Y

def main():
    try:
        X, Y = load_datasets()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    regression_engine(X, Y)


if __name__ == "__main__":
    main()