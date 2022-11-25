import sys
sys.path.append('../')
from utils.utils import *
from ex06.ridge import MyRidge
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ex07.benchmark_train import normalize

def unormalize(n_X, X):
    mean_X = np.mean(X)
    std_X = np.std(X)
    return n_X * std_X + mean_X

def load_models():
    file = open('models.pickle', 'rb')
    models = pickle.load(file)
    file.close()
    return models

def load_dataset():
    content = pd.read_csv("space_avocado.csv")
    X = np.array(content[["weight", "prod_distance", "time_delivery"]])
    if X.shape[1] !=  3:
        raise Exception("Datas are missing in space_avocado.csv")        
    Y = np.array(content[["target"]])
    if Y.shape[1] !=  1:
        raise Exception("Datas are missing in space_avocado.csv")   
    return X, Y

def data_cross_splitter(x, y):
    perm = np.random.permutation(len(x))
    s_x = x[perm]
    s_y = y[perm]
    x_cross = np.split(s_x, 10)
    y_cross = np.split(s_y, 10)
    return x_cross, y_cross


def compare_mses(mses):
    plt.rcParams["figure.figsize"] = (20,7)
    for mse in mses:
        plt.scatter(mse, mses[mse], label=mse)
    plt.grid()
    plt.xticks(rotation="vertical")
    plt.xlabel("models")
    plt.ylabel("mse")
    plt.show()

def cross_evaluate_models(x, y, models):
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
    x_train = add_polynomial_features(x_train, 4)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    #x_cross, y_cross = data_cross_splitter(x, y)
    #x_cross = normalize(x_cross)
    #y_cross = normalize(y_cross)
    best_mse = -1
    best_model = None
    mses = {}
    for model in models:
        myLR =  models[model]
        w_features = [0, 3, 6, 9][:int(model[1])]
        d_features = [1, 4, 7, 10][:int(model[3])]
        t_features = [2, 5, 8, 11][:int(model[5])]
        x_features = x_train[:, np.concatenate((w_features, d_features, t_features))]
        y_pred = myLR.predict_(x_features)
        current_mse = MyRidge.mse_(y_train, y_pred)
        mses[model] = current_mse
        if best_mse < 0 or current_mse < best_mse:
            best_mse = current_mse
            best_model = model
    print(best_model, best_mse)
    compare_mses(mses)
    return best_model, models[best_model]

def compare_pred(pred, X, Y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Predictions comparisions")
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.scatter(X[:,0], pred, label="Predictions")
    ax1.legend()
    ax1.set_xlabel("weight")
    ax1.set_ylabel("price")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.scatter(X[:,1], pred, label="Predictions")
    ax2.legend()
    ax2.set_xlabel("prod_distance")
    ax2.set_ylabel("price")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.scatter(X[:,2], pred, label="Predictions")
    ax3.legend()
    ax3.set_xlabel("time_delivery")
    ax3.set_ylabel("price")
    fig.tight_layout()
    plt.show()

def train_model(myLR, model, x, y):
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
    x_train = add_polynomial_features(x_train, 4)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    w_features = [0, 3, 6, 9][:int(model[1])]
    d_features = [1, 4, 7, 10][:int(model[3])]
    t_features = [2, 5, 8, 11][:int(model[5])]
    x_features = x_train[:, np.concatenate((w_features, d_features, t_features))]
    myLR.fit_(x_features, y_train)
    x_test_features = normalize(add_polynomial_features(x_test, 4))
    preds = myLR.predict_(x_test_features[:,np.concatenate((w_features, d_features, t_features)) ])
    preds = unormalize(preds, y_test)
    compare_pred(preds, x_test, y_test)

def main():
    try:
        x, y = load_dataset()
        models = load_models()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    best_model, model = cross_evaluate_models(x, y, models)
    train_model(model, best_model, x, y)

if __name__ == "__main__":
    main()