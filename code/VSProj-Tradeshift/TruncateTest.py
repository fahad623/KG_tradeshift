import pandas as pd
import numpy as np

totalSize = 545082
sizeToKeep = 50000
size = totalSize - sizeToKeep

skiprows = np.random.choice(np.arange(1, totalSize), size = size, replace=False)
skiprows.sort(axis=0)

df_test_X = pd.read_csv("..\\..\\data\\test.csv", skiprows = skiprows)
df_test_X.to_csv("..\\..\\data\\trunc_test.csv", index = False)