import pandas as pd

clfFolder = "..\\..\\classifier\\LinearSVC_DF\\"

df_base1 = pd.read_csv(clfFolder + "output_base_dec_func_part1.csv")
print df_base1.shape
df_base2 = pd.read_csv(clfFolder + "output_base_dec_func_part2.csv")
print df_base2.shape
df_base2.drop(['id'], axis=1, inplace = True)

df_output = df_base1.join(df_base2)
print df_output.shape
df_output.to_csv(clfFolder + "output_base_dec_func.csv", index = False)