import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import *
from ex08.my_logistic_regression import *
import matplotlib.pyplot as plt
import pickle


def load_models():
    file = open('zscore-models.pickle', 'rb')
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
    plt.ylabel("f1score")
    plt.show()

def evaluate_models(models_data):
    models = models_data["models"]
    models_score = models_data["models_score"]
    best_model = None
    for model_score in models_score:
        print(f"{model_score} : {models_score[model_score]}" )
        if not best_model or models_score[model_score] > models_score[best_model]:
            best_model = model_score
    print(f"best one => {best_model} : {models_score[best_model]}")
    compare_mses(model_score)
    return best_model

def compare_pred(preds, X, Y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Predictions comparisions")
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.set_xlabel("weight")
    ax1.set_ylabel("price")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.set_xlabel("prod_distance")
    ax2.set_ylabel("price")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.set_xlabel("time_delivery")
    ax3.set_ylabel("price")
    for model in preds:   
        ax1.scatter(X[:,0], preds[model], label=model)
        ax2.scatter(X[:,1], preds[model], label=model)
        ax3.scatter(X[:,2], preds[model], label=model)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.tight_layout()
    plt.show()

def train_model(myR, model, x, y):
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
    x_train = add_polynomial_features(x_train, 4)
    x_train = normalize(x_train)
    w_features = [0, 3, 6, 9][:int(model[1])]
    d_features = [1, 4, 7, 10][:int(model[3])]
    t_features = [2, 5, 8, 11][:int(model[5])]
    x_train_features = x_train[:, np.concatenate((w_features, d_features, t_features))]
    x_test_features = normalize(add_polynomial_features(x_test, 4))
    params = myR.get_params_()
    params["thetas"] = np.random.rand(x_train_features.shape[1] + 1, 1).reshape(-1, 1)
    l_preds = {}
    for l, lambda_ in enumerate([0., 0.2, 0.4, 0.6, 0.8]):
        model_name = f"{model[:6]}λ{lambda_}"
        params["lambda_"] = lambda_
        myR.set_params_(params)
        myR.fit_(x_train_features, y_train)
        preds = myR.predict_(x_test_features[:,np.concatenate((w_features, d_features, t_features)) ])
        l_preds[model_name] = preds
    compare_pred(l_preds, x_test, y_test)

def main():
    try:
        x, y = load_dataset()
        models_data = load_models()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    models_score = models_data["models_score"]
    models = models_data["models"]
    best_model = evaluate_models(models_score)
    train_model(models[best_model], best_model, x, y)

if __name__ == "__main__":
    main()