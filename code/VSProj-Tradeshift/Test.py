import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

train = pd.DataFrame(dict(col1 = ['hgf', 'kjh', 'ijk'], col2 = ['kjh', '123', '123'], col3 = [1.3, 4, 5]))

train_as_dicts = [dict(r.iteritems()) for _, r in train.iterrows()]

print train.dtypes


vectorizer = DictVectorizer()
vectorized_sparse = vectorizer.fit_transform(train_as_dicts)

vectorized_array = vectorized_sparse.toarray()

print vectorized_array
print vectorizer.get_feature_names()