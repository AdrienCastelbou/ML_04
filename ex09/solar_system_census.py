import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import *
from ex08.my_logistic_regression import *
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d


def load_models():
    file = open('models.pickle', 'rb')
    models = pickle.load(file)
    file.close()
    return models

def load_datasets():
    content = pd.read_csv("solar_system_census.csv")
    X = np.array(content[["height", "weight", "bone_density"]])
    if X.shape[1] !=  3:
        raise Exception("Datas are missing in solar_system_census.csv")        
    content = pd.read_csv("solar_system_census_planets.csv")
    Y = np.array(content[["Origin"]])
    if Y.shape[1] !=  1:
        raise Exception("Datas are missing in solar_system_census_planets.csv")   
    return X, Y

def data_cross_splitter(x, y):
    perm = np.random.permutation(len(x))
    s_x = x[perm]
    s_y = y[perm]
    x_cross = np.split(s_x, 10)
    y_cross = np.split(s_y, 10)
    return x_cross, y_cross


def compare_f1_scores_(models_score):
    plt.rcParams["figure.figsize"] = (20,7)
    for model_score in models_score:
        plt.scatter(model_score, models_score[model_score], label=model_score)
    plt.grid()
    plt.xticks(rotation="vertical")
    plt.xlabel("models")
    plt.ylabel("f1score")
    plt.show()

def evaluate_models(models_score):
    best_model = None
    for model_score in models_score:
        print(f"{model_score} : {models_score[model_score]}" )
        if not best_model or models_score[model_score] > models_score[best_model]:
            best_model = model_score
    print(f"best one => {best_model} : {models_score[best_model]}")
    #compare_f1_scores_(models_score)
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


def binarize(Y_train, reference):
    bin_ = np.zeros(Y_train.shape)
    for i in range(bin_.shape[0]):
        if reference == float(Y_train[i]):
            bin_[i] = 1.
        else:
            bin_[i] = 0.
    return bin_

def train_model(X_train, Y_train, lambda_):
    classifiers = []
    for i in range(4):
        myLR = MyLogisticRegression(theta=np.random.rand(X_train.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-1, max_iter=10000, lambda_=lambda_)
        myLR.fit_(X_train, binarize(Y_train, i))
        classifiers.append(myLR)
    return classifiers



def perform_one_vs_all(model, x, y):
    preds = []
    for classifier in model:
        preds.append(classifier.predict_(x))
    y_hat = np.zeros(y.shape)
    for i, cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred in zip(range(y_hat.shape[0]), preds[0], preds[1], preds[2], preds[3]):
        best = max(cl_zero_pred, cl_one_pred, cl_two_pred, cl_three_pred)
        if best == cl_zero_pred:
            y_hat[i] = 0
        elif best == cl_one_pred:
            y_hat[i] = 1
        elif best == cl_two_pred:
            y_hat[i] = 2
        elif best == cl_three_pred:
            y_hat[i] = 3
    print(f'Precision : {len(y_hat[y_hat == y])} / {len(y_hat)}')
    f1score = multiclass_f1_score_(y, y_hat, [0, 1, 2, 3])
    print("f1score :" , f1score)
    return y_hat

def vizualize_predictions(x, y, y_hat):
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Solar system census Origins vs Predictions")
    scatter = ax.scatter(x[:,0],x[:,1],x[:,2], c=y, cmap="gist_rainbow", s=50) # plot the point (2,3,4) on the figure
    ax.scatter(x[:,0],x[:,1],x[:,2], c=y_hat, cmap="gist_rainbow", s=26) # plot the point (2,3,4) on the figure
    ax.set_xlabel("Height")
    ax.set_ylabel("weight")
    ax.set_zlabel("bone density")
    ax.legend(*scatter.legend_elements(), loc="lower left", title="Origins")
    plt.show()

def perform_regression(best_model, x, y):
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)

    x_train_tr = add_polynomial_features(x_train, 4)
    x_train_tr = normalize(x_train_tr)
    x_test_tr = normalize(add_polynomial_features(x_test, 4))
    w_features = [0, 3, 6, 9][:int(best_model[1])]
    d_features = [1, 4, 7, 10][:int(best_model[3])]
    t_features = [2, 5, 8, 11][:int(best_model[5])]
    lambda_ = 0.
    x_train_features = x_train_tr[:, np.concatenate((w_features, d_features, t_features))]
    x_test_features = x_test_tr[:, np.concatenate((w_features, d_features, t_features))]

    model = train_model(x_train_features, y_train, lambda_)
    preds = perform_one_vs_all(model, x_test_features, y_test)
    vizualize_predictions(x_test, y_test, preds)

def main():
    try:
        x, y = load_datasets()
        models_data = load_models()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    models_score = models_data["models_score"]
    for e in models_data:
        print(e)
    models = models_data['models']
    best_model = evaluate_models(models_score)
    model = perform_regression(best_model, x, y)

if __name__ == "__main__":
    main()