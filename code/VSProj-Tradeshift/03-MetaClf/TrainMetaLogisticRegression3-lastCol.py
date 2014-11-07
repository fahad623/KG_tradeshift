import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
import os
import shutil

trainFileY = "..\\..\\..\\data\\trainLabels.csv"

clfFolder_base1 = "..\\..\\..\\classifier\\RandomForest\\"
clfFolder_base2 = "..\\..\\..\\classifier\\LinearSVC_DF\\"
clfFolder = "..\\..\\..\\classifier\\Meta-lastCol\\"


def cv_optimize(X_train, Y_train, clf):
    alpha_range = 10.0 ** np.arange(-6, 3)
    param_grid = dict(alpha=alpha_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 10, n_jobs = 8, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = SGDClassifier(loss = 'log', n_iter = 300, shuffle = True, random_state = 49)
    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf

if __name__ == '__main__':
	
    df_train_X1 = pd.read_csv(clfFolder_base1 + "output_base_proba.csv")
    df_train_X2 = pd.read_csv(clfFolder_base2 + "output_base_dec_func.csv")
    df_train_Y = pd.read_csv(trainFileY)
    yCols = df_train_Y.columns.values.tolist()

    df_train_output = pd.DataFrame(df_train_Y[['id']])
    df_train_output_proba = pd.DataFrame(df_train_Y[['id']])

    for colName in yCols[-1:]:

        Y_train = df_train_Y[colName].values
        if np.any(Y_train != 0):
            X_train1 = df_train_X1[colName].values
            X_train2 = df_train_X2[colName].values
            X_train = np.vstack((X_train1, X_train2)).T
            print X_train.shape
            del X_train1, X_train2
            clf = fit_clf(X_train, Y_train)
            df_train_output[colName] = clf.predict(X_train)
            df_train_output_proba[colName] = clf.predict_proba(X_train)[:, 1]

            pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)

            if not os.path.exists(pathClassifier):
                os.makedirs(pathClassifier)

            score_file = open(pathClassifier + "Score.txt", "w")
            score_file.write("Score = {0}".format(clf.score(X_train, Y_train)))
            score_file.close()
            joblib.dump(clf, pathClassifier+'model.pkl')
        else:
            df_train_output[colName] = 0
            df_train_output_proba[colName] = 0

    df_train_output.to_csv(clfFolder + "output_meta.csv", index = False)
    df_train_output_proba.to_csv(clfFolder + "output_meta_proba.csv", index = False)