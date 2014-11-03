import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, train_test_split, KFold

clfFolder = "..\\..\\..\\classifier\\RandomForest\\"

trainFileX = "..\\..\\..\\data\\train.csv"
trainFileY = "..\\..\\..\\data\\trainLabels.csv"

def clean_features(df):
    df.replace('YES', 1, inplace = True)
    df.replace('NO', 0, inplace = True)
    df.replace('nan', np.NaN, inplace = True)

    colsDrop = ['x3', 'x4', 'x34', 'x35', 'x61', 'x64', 'x65', 'x91', 'x94', 'x95']
    df.drop(colsDrop, axis=1, inplace = True)

    df.fillna(-999, inplace = True)


if __name__ == '__main__':

    df_train_X = pd.read_csv(trainFileX)
    clean_features(df_train_X)

    df_train_Y = pd.read_csv(trainFileY)

    yCols = df_train_Y.columns.values.tolist()
    
    df_output = pd.DataFrame(df_train_Y[['id']])

    for colName in yCols[1:]:
        
        pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)        

        kf = KFold(df_train_X.shape[0], n_folds = 10)

        for train, test in kf:

            if os.path.exists(pathClassifier):

                df_trainx_split = df_train_X.loc[train]
                df_testx_split = df_train_X.loc[test]

                df_trainy_split = df_train_Y.loc[train]
                df_testy_split = df_train_Y.loc[test]

                X_train = df_trainx_split.values[:, 1:]
                X_test  = df_testx_split.values[:, 1:]
                Y_train = df_trainy_split[colName].values
                
                clf = joblib.load(pathClassifier+'model.pkl')
                clf.fit(X_train, Y_train)

                df_output.loc[df_testx_split.index.values, colName] = clf.predict(X_test)
            else:
                df_output[colName] = 0

    df_output.to_csv(clfFolder + "output_base.csv", index = False) 

