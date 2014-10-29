import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn.externals import joblib

if __name__ == '__main__':

    df_train_X = pd.read_csv("..\\..\\data\\trunc_train.csv")
    df_train_X.replace('YES', 1, inplace = True)
    df_train_X.replace('NO', 0, inplace = True)
    df_train_X.replace('nan', np.NaN, inplace = True)
    print df_train_X.shape

    df_train_X1 = df_train_X.dropna(axis=0)
    print df_train_X1.shape
    del df_train_X

    df_train_Y = pd.read_csv("..\\..\\data\\trunc_trainLabels.csv")
    print df_train_Y.shape
    df_train_Y1 = df_train_Y.loc[df_train_X1.index.values]
    print df_train_Y1.shape
    
    X_train = df_train_X.values[:, 1:]

    yCols = df_train_Y.columns.values.tolist()

    for colName in yCols[1:2]:
        Y_train = df_train_Y[colName].values

        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, Y_train)

        joblib.dump(clf, '..\\..\\classifier\\my_model.pkl')
        