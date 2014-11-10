import pandas as pd

clfFolder = "..\\..\\..\\classifier\\LinearSVC_DF\\"

df_input = pd.read_csv(clfFolder + "output_base_dec_func.csv")
print df_input.shape

def apply_func(row):
    print row.id
    row_short = row.iloc[1:33]
    row_mask_more = (row_short > 0)
    if pd.Series.any(row_mask_more):
        row['y33'] = 1
    else:        
        row['y33'] = (1 - row_short.mean())
    
    return row

df_input = df_input.apply(apply_func, axis=1)
df_input['id'] = df_input['id'].astype(np.int64, copy=False)
print df_input.head()

df_input.to_csv(clfFolder + "output_base_dec_func41.csv", index = False)