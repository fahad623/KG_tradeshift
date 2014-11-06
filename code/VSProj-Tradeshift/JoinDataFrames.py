import pandas as pd

clfFolder = "..\\..\\classifier\\LinearSVC\\"

df_base1 = pd.read_csv(clfFolder + "output_base_part1.csv")
df_base2 = pd.read_csv(clfFolder + "output_base_part2.csv")

df_output = df_base1.join(df_base2)

df_output.to_csv(clfFolder + "output_base.csv", index = False)