import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib


clfFolder_base1 = "..\\..\\..\\classifier\\RandomForest\\"
clfFolder_base2 = "..\\..\\..\\classifier\\LinearSVC\\"
clfFolder = "..\\..\\..\\classifier\\Meta\\"


if __name__ == '__main__':

    df_test_X1 = pd.read_csv(clfFolder_base1 + "output_proba.csv")
    df_test_X2 = pd.read_csv(clfFolder_base2 + "output.csv")

    df_output = pd.DataFrame(df_test_X1[['id']])
    df_output_proba = pd.DataFrame(df_test_X1[['id']])

    yCols = df_test_X1.columns.values.tolist()

    for colName in yCols[1:]:

        print "Working on " + colName
        
        pathClassifier = clfFolder+'classifier_{0}\\'.format(colName)  

        if os.path.exists(pathClassifier):

            X_test1 = df_test_X1[colName].values
            X_test2 = df_test_X2[colName].values
            X_test = np.vstack((X_test1, X_test2)).T

            clf = joblib.load(pathClassifier+'model.pkl')
            df_output_proba[colName] = clf.predict_proba(X_test)[:, 1]
            df_output[colName] = clf.predict(X_test)
        else:
            df_output[colName] = 0
            df_output_proba[colName] = 0

    df_output.to_csv(clfFolder + "output.csv", index = False)
    df_output_proba.to_csv(clfFolder + "output_proba.csv", index = False)

