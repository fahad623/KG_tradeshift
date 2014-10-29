import pandas as pd
import numpy as np

#df = pd.DataFrame(dict(col1=[1,2,3,4,5], col2 = [3,4,5,6,7]))
#rows = np.random.choice(df.index.values, 2, replace=False)
#print df.ix[rows]

#df_train_X = pd.read_csv("..\\..\\data\\train.csv")
#rows = np.random.choice(df_train_X.index.values, 50000, replace=False)
#df_train_X_trunc = df_train_X.ix[rows]
#df_train_X_trunc.to_csv("..\\..\\data\\trunc_train.csv", index = False)

#del df_train_X_trunc
#del df_train_X

#df_train_Y = pd.read_csv("..\\..\\data\\trainLabels.csv")
#df_train_Y_trunc = df_train_Y.ix[rows]
#df_train_Y_trunc.to_csv("..\\..\\data\\trunc_trainLabels.csv", index = False)

totalSize = 1700000
sizeToKeep = 50000
size = totalSize - sizeToKeep

skiprows = np.random.choice(np.arange(1, totalSize), size = size, replace=False)
skiprows.sort(axis=0)

df_train_X = pd.read_csv("..\\..\\data\\train.csv", skiprows = skiprows)
df_train_X.to_csv("..\\..\\data\\trunc_train.csv", index = False)

del df_train_X

df_train_Y = pd.read_csv("..\\..\\data\\trainLabels.csv", skiprows = skiprows)
df_train_Y.to_csv("..\\..\\data\\trunc_trainLabels.csv", index = False)

#totalSize = 1700
#sizeToKeep = 500
#size = totalSize - sizeToKeep
#skiprows = np.random.choice(np.arange(totalSize), size = size, replace=False)
#skiprows.sort(axis=0)
#print skiprows.shape

#df_train_X = pd.read_csv("..\\..\\datatest\\sample_submission.csv", skiprows = skiprows)
#df_train_X.to_csv("..\\..\\datatest\\trunc_train.csv", index = False)