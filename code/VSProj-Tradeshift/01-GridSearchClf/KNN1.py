import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.grid_search import GridSearchCV
import os
import shutil

trainFileX = "..\\..\\..\\data\\trunc_train.csv"
trainFileY = "..\\..\\..\\data\\trunc_trainLabels.csv"
testFileX = "..\\..\\..\\data\\trunc_test.csv"
clfFolder = "..\\..\\..\\classifier\\NearestNeighbors\\"

def cv_optimize(X_train, Y_train, clf):
    n_neighbors_range = np.arange(5, 51, 5)
    param_grid = dict(n_neighbors = n_neighbors_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 10, n_jobs = 8, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = KNeighborsClassifier()

    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf

def clean_features(df):
    df.replace('YES', 1, inplace = True)
    df.replace('NO', 0, inplace = True)
    df.replace('nan', np.NaN, inplace = True)

    colsDrop = ['x3', 'x4', 'x34', 'x35', 'x61', 'x64', 'x65', 'x91', 'x94', 'x95']
    df.drop(colsDrop, axis=1, inplace = True)

    df.fillna(-999, inplace = True)

if __name__ == '__main__':

    shutil.rmtree(clfFolder, ignore_errors=True)

    df_train_X = pd.read_csv(trainFileX)
    clean_features(df_train_X)

    df_train_Y = pd.read_csv(trainFileY)
    
    X_train = df_train_X.values[:, 1:]
    yCols = df_train_Y.columns.values.tolist()

    df_test = pd.read_csv(testFileX)
    clean_features(df_test)
    X_test = df_test.values[:, 1:]

    df_output = pd.DataFrame(df_test[['id']])
    df_output_proba = pd.DataFrame(df_test[['id']])

    for colName in yCols[1:]:
        Y_train = df_train_Y[colName].values

        if np.any(Y_train != 0):
            clf = fit_clf(X_train, Y_train)

            predicted_test_proba = clf.predict_proba(X_test)
            df_output_proba[colName] = predicted_test_proba[:, 1]

            df_output[colName] = clf.predict(X_test)
            df_train_Y[colName] = clf.predict(X_train)

            pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)

            if not os.path.exists(pathClassifier):
                os.makedirs(pathClassifier)

            score_file = open(pathClassifier+"Score.txt", "w")
            score_file.write("Score = {0}".format(clf.score(X_train, Y_train)))
            score_file.close()
            joblib.dump(clf, pathClassifier+'model.pkl')
        else:
            df_output_proba[colName] = 0
            df_output[colName] = 0
            df_train_Y[colName] = 0

    df_output_proba.to_csv(clfFolder + "output_proba.csv", index = False)
    df_output.to_csv(clfFolder + "output.csv", index = False) 
    df_train_Y.to_csv(clfFolder + "y_predict.csv", index = False)