import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import *
from ex08.my_logistic_regression import *
import matplotlib.pyplot as plt
import pickle

def data_cross_splitter(x, y, k=10):
    perm = np.random.permutation(len(x))
    s_x = x[perm]
    s_y = y[perm]
    x_cross = np.array_split(s_x, k)
    y_cross = np.array_split(s_y, k)
    return x_cross, y_cross

def save_models(results):
    file = open('cross_models.pickle', 'wb')
    pickle.dump(results, file)
    file.close()

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


def vizualize_preds(X, Y, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12.5, 5.5)
    fig.tight_layout()
    ax1.scatter(X[:,0], Y, label="Real values")
    ax1.scatter(X[:,0], pred, label="Predictions")
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel("weight")
    ax1.set_ylabel("Origin")
    ax2.scatter(X[:,1], Y, label="Real values")
    ax2.scatter(X[:,1], pred, label="Predictions")
    ax2.grid()
    ax2.legend()
    ax2.set_xlabel("height")
    ax2.set_ylabel("Origin")
    ax3.scatter(X[:,2], Y, label="Real values")
    ax3.scatter(X[:,2], pred, label="Predictions")
    ax3.grid()
    ax3.legend()
    ax3.set_xlabel("bone_density")
    ax3.set_ylabel("Origin")
    fig.suptitle("Predictions comparisions")
    plt.show()

def evaluate_model(model, x, y):
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
    return f1score

def binarize(Y_train, reference):
    bin_ = np.zeros(Y_train.shape)
    for i in range(bin_.shape[0]):
        if reference == float(Y_train[i]):
            bin_[i] = 1.
        else:
            bin_[i] = 0.
    return bin_

def train_model(x, y, lambda_):
    x_cross, y_cross = data_cross_splitter(x, y, 4)
    f1_scores = []
    for k in range(len(x_cross)):
        print(f"fold {k}")
        x_folds_test = np.concatenate([x_fold for i, x_fold in enumerate(x_cross) if i!=k])
        x_fold_evaluate = x_cross[k]
        y_folds_test = np.concatenate([y_fold for i, y_fold in enumerate(y_cross) if i!=k])
        y_fold_evaluate = y_cross[k]
        classifiers = []
        for i in range(4):
            myLR = MyLogisticRegression(theta=np.random.rand(x.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-1, max_iter=10000, lambda_=lambda_)
            myLR.fit_(x_folds_test, binarize(y_folds_test, i))
            classifiers.append(myLR)
        f1_scores.append(evaluate_model(classifiers, x_fold_evaluate, y_fold_evaluate))
    return classifiers, np.mean(f1_scores)

def perform_multi_classification(X, Y):
    X_train = add_polynomial_features(X, 3)
    X_train = normalize(X_train)
    models = {}
    models_score = {}
    for h_rank in range(1, 4): 
        h_features = [0, 3, 6][:h_rank]
        for w_rank in range(1, 4):
            w_features = [1, 4, 7][:w_rank]
            for b_rank in range(1, 4):
                b_features = [2, 5, 8, 11][:b_rank]
                X_train_features = X_train[:, np.concatenate((h_features, w_features, b_features))]
                for l, lambda_ in enumerate([0., 0.2, 0.4, 0.6, 0.8]):
                    classifiers_rank = f"w{h_rank}d{w_rank}t{b_rank}λ{lambda_}"
                    print(classifiers_rank)
                    models[classifiers_rank], models_score[classifiers_rank] = train_model(X_train_features, Y, lambda_)  
                    print("F1 score mean : ", models_score[classifiers_rank])                  
    save_models({"models_score": models_score, "models": models})
    
        

def main():
    try:
        X, Y = load_datasets()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    perform_multi_classification(X, Y)   
    

if __name__ == "__main__":
    main()