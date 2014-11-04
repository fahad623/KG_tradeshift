import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import os
import shutil

trainFileX = "..\\..\\..\\data\\train.csv"
trainFileY = "..\\..\\..\\data\\trainLabels.csv"
testFileX = "..\\..\\..\\data\\test.csv"
clfFolder = "..\\..\\..\\classifier\\LinearSVC-lastCol2\\"

def cv_optimize(X_train, Y_train, clf):
    C_range = [0.1, 1, 10, 100]
    param_grid = dict(C=C_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 5, n_jobs = 2, verbose = 3)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = LinearSVC(C = 0.1)
    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf

def clean_features(df):
    df.replace('YES', 1, inplace = True)
    df.replace('NO', 0, inplace = True)
    df.replace('nan', np.NaN, inplace = True)

    colsDrop = ['x3', 'x4', 'x34', 'x35', 'x61', 'x64', 'x65', 'x91', 'x94', 'x95']
    df.drop(colsDrop, axis=1, inplace = True)

    df.fillna(0, inplace = True)

def normalize_data(df_train, df_test):
    exclude = ['x1', 'x2', 'x10', 'x11', 'x12', 'x13', 'x14', 'x24', 'x25', 'x26', 'x30', 'x31', 'x32', 'x33', 
               'x41', 'x42', 'x43', 'x44', 'x45', 'x55', 'x56', 'x57', 'x58', 'x62', 'x63', 'x71', 'x72', 'x73',
               'x74', 'x75', 'x85', 'x86', 'x87', 'x92', 'x93', 'x101', 'x102', 'x103', 'x104', 'x105', 'x115',
               'x116', 'x117', 'x126', 'x127', 'x128', 'x129', 'x130',  'x140', 'x141', 'x142']
    include = [item for item in df_train.columns.values[1:] if item not in exclude]

    train_exclude = df_train[exclude].values
    train_include = df_train[include].values

    test_exclude = df_test[exclude].values
    test_include = df_test[include].values

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(train_include)
    X_test = scaler.transform(test_include)

    X_train_full = np.hstack((train_exclude, train_include))
    X_test_full = np.hstack((test_exclude, test_include))

    del train_exclude
    del train_include
    del test_exclude
    del test_include
    del X_train
    del X_test

    return X_train_full, X_test_full

if __name__ == '__main__':

    shutil.rmtree(clfFolder, ignore_errors=True)

    df_train_X = pd.read_csv(trainFileX)
    clean_features(df_train_X)

    df_test = pd.read_csv(testFileX)
    clean_features(df_test)

    X_train, X_test = normalize_data(df_train_X, df_test) 

    df_train_Y = pd.read_csv(trainFileY)
    yCols = df_train_Y.columns.values.tolist()

    df_output = pd.DataFrame(df_test[['id']])

    del df_train_X
    del df_test

    for colName in yCols[-1:]:
        print colName
        Y_train = df_train_Y[colName].values

        if np.any(Y_train != 0):
            clf = fit_clf(X_train, Y_train)

            df_output[colName] = clf.predict(X_test)

            pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)

            if not os.path.exists(pathClassifier):
                os.makedirs(pathClassifier)

            joblib.dump(clf, pathClassifier+'model.pkl')
        else:
            df_output[colName] = 0

    df_output.to_csv(clfFolder + "output.csv", index = False)