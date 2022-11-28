import sys
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.utils import *
from ex08.my_logistic_regression import MyLogisticRegression
import matplotlib.pyplot as plt

def save_models(results):
    file = open('zscore-models.pickle', 'wb')
    pickle.dump(results, file)
    file.close()

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

def engine_Y(Y_train, reference):
    for i in range(Y_train.shape[0]):
        if reference == float(Y_train[i]):
            Y_train[i] = 1.
        else:
            Y_train[i] = 0.
    return Y_train

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


def evaluate_model(classifiers, X, Y):
    preds = []
    for classifier in classifiers:
        preds.append(classifier.predict_(X))
    y_hat = np.zeros(Y.shape)
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
    print(f'Precision : {len(y_hat[y_hat == Y])} / {len(y_hat)}')
    vizualize_preds(X , Y, y_hat)

def train_classifiers(X_train, Y_train, lambda_):
    X_cross_train, Y_cross_train = data_cross_splitter(X_train, Y_train, 5)
    X_cross_train = X_train
    Y_cross_train = Y_train
    classifiers = []
    f1scores = []
    #for validation_fold_idx in range(len(X_cross_train)):
    #    X_train_fold = np.concatenate([x for i,x in enumerate(X_cross_train) if i!=validation_fold_idx])
    ##    X_validation_fold = X_cross_train[validation_fold_idx]
     #   Y_train_fold = np.concatenate([x for i,x in enumerate(Y_cross_train) if i!=validation_fold_idx])
    #    Y_validation_fold = Y_cross_train[validation_fold_idx]
    for label in range(4):
            Y_train = engine_Y(np.copy(Y_cross_train), label)
            myLR = MyLogisticRegression(theta=np.random.rand(X_train.shape[1] + 1, 1).reshape(-1, 1), max_iter=15000, lambda_=lambda_)
            myLR.fit_(X_cross_train, Y_cross_train)
            classifiers.append(myLR)
    return classifiers
    f1scores.append(evaluate_model(classifiers, X_cross_train, Y_cross_train))



def perform_multi_classification(X, Y):
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.8)
    X_train = add_polynomial_features(X_train, 4)
    X_train = normalize(X_train)
    classifiers = {}
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
                    classifiers[classifiers_rank] = train_classifiers(X_train_features, Y_train, lambda_)
                    break
    X_test = normalize(X_test)
    for classifier in classifiers:
        print(classifiers[classifier])
        evaluate_model(classifiers[classifier], X_test, Y_test)
                

def main():
    try:
        X, Y = load_datasets()
    except Exception as e:
        print("Error in datas loading :", e)
        return
    perform_multi_classification(X, Y)   
    

if __name__ == "__main__":
    main()