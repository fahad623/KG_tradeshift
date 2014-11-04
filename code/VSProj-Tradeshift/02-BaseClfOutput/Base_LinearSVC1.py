import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, train_test_split, KFold

clfFolder = "..\\..\\..\\classifier\\LinearSVC\\"

trainFileX = "..\\..\\..\\data\\train.csv"
trainFileY = "..\\..\\..\\data\\trainLabels.csv"

def normalize_data(df_train):
    exclude = ['x1', 'x2', 'x10', 'x11', 'x12', 'x13', 'x14', 'x24', 'x25', 'x26', 'x30', 'x31', 'x32', 'x33', 
               'x41', 'x42', 'x43', 'x44', 'x45', 'x55', 'x56', 'x57', 'x58', 'x62', 'x63', 'x71', 'x72', 'x73',
               'x74', 'x75', 'x85', 'x86', 'x87', 'x92', 'x93', 'x101', 'x102', 'x103', 'x104', 'x105', 'x115',
               'x116', 'x117', 'x126', 'x127', 'x128', 'x129', 'x130',  'x140', 'x141', 'x142']
    include = [item for item in df_train.columns.values[1:] if item not in exclude]

    train_exclude = df_train[exclude].values
    train_include = df_train[include].values

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(train_include)

    X_train_full = np.hstack((train_exclude, train_include))

    del train_exclude
    del train_include
    del X_train

    return X_train_full

def clean_features(df):
    df.replace('YES', 1, inplace = True)
    df.replace('NO', 0, inplace = True)
    df.replace('nan', np.NaN, inplace = True)

    colsDrop = ['x3', 'x4', 'x34', 'x35', 'x61', 'x64', 'x65', 'x91', 'x94', 'x95']
    df.drop(colsDrop, axis=1, inplace = True)

    df.fillna(0, inplace = True)


if __name__ == '__main__':

    df_train_X = pd.read_csv(trainFileX)
    clean_features(df_train_X)
    X_train_full = normalize_data(df_train_X)

    df_train_Y = pd.read_csv(trainFileY)
    yCols = df_train_Y.columns.values.tolist()
    df_output = pd.DataFrame(df_train_Y[['id']])

    for colName in yCols[1:]:

        print "Working on "+colName
        
        pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)        

        kf = KFold(X_train.shape[0], n_folds = 10)

        for train, test in kf:

            if os.path.exists(pathClassifier):

                df_trainx_split = df_train_X.loc[train]
                df_testx_split = df_train_X.loc[test]

                df_trainy_split = df_train_Y.loc[train]
                df_testy_split = df_train_Y.loc[test]

                X_train = X_train_full[train]
                X_test  = X_train_full[test]
                Y_train = df_trainy_split[colName].values
                
                clf = joblib.load(pathClassifier+'model.pkl')
                clf.fit(X_train, Y_train)

                df_output.loc[df_testx_split.index.values, colName] = clf.predict(X_test)

                del df_trainx_split, df_testx_split, df_trainy_split, df_testy_split, X_train, X_test, Y_train, clf
            else:
                df_output[colName] = 0

    df_output.to_csv(clfFolder + "output_base.csv", index = False)

