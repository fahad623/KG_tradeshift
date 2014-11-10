import pandas as pd

clfFolder = "..\\..\\..\\classifier\\Meta\\"

df_input = pd.read_csv(clfFolder + "output_proba.csv")
print df_input.shape
df_output = pd.DataFrame(df_input)
df_input.drop(['id', 'y33'], axis=1, inplace = True)

for index, row in df_input.iterrows():
    print index
    row_mask_more = (row >= 0.5)
    if pd.Series.any(row_mask_more):
        df_output.loc[index,'y33'] = 1 - row[row_mask_more].mean()
    else:        
        df_output.loc[index,'y33'] = 1 - row.mean()

df_output.to_csv(clfFolder + "output_proba2.csv", index = False)

