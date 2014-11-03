import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

#train = pd.DataFrame(dict(col1 = ['hgf', 'kjh', 'ijk'], col2 = ['kjh', '123', '123'], col3 = [1.3, 4, 5]))

#train['col1'] = 0

#print train

#train_as_dicts = [dict(r.iteritems()) for _, r in train.iterrows()]

#print train.dtypes


#vectorizer = DictVectorizer()
#vectorized_sparse = vectorizer.fit_transform(train_as_dicts)

#vectorized_array = vectorized_sparse.toarray()

#print vectorized_array
#print vectorizer.get_feature_names()

#df = pd.read_csv("..\\..\\output_final.csv")
#print df.shape
#del df
#df = pd.read_csv("..\\..\\data\\test.csv")
#print df.shape
#del df
#df = pd.read_csv("..\\..\\classifier\\RandomForest\\output_proba.csv")
#print df.shape
#del df

from sklearn.cross_validation import cross_val_score, train_test_split, KFold

#trainX = [[1,2,3,4],
#          [2,3,4,5],
#          [6,2,3,4],
#          [7,3,4,5],
#          [8,2,3,4],
#          [9,3,4,5]]

#trainY = [[1],
#          [2],
#          [3],
#          [4],
#          [5],
#          [6]]

#df = pd.DataFrame(dict(id = [13, 14, 15, 16, 17, 18],col1=[1,2,3,4,5,6], col2 = [5,6,7,8,9,1]))
#df_output = pd.DataFrame(df[['id']])

#kf = KFold(df.shape[0], n_folds=3)
#for train, test in kf:
#    df_train = df.loc[train]
#    df_test = df.loc[test]
#    df_output.loc[df_test.index.values,'out1'] = 0

#    print df_train
#    print df_test
#    print df_output

trainFileX = "..\\..\\data\\trunc_train.csv"
df1 = pd.read_csv(trainFileX)
df2 = pd.read_csv(trainFileX)
df3 = pd.read_csv(trainFileX)
df4 = pd.read_csv(trainFileX)
df5 = pd.read_csv(trainFileX)

del df1
del df2
del df3
a = 9

