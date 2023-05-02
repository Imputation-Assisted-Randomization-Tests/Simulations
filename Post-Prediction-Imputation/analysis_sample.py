import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("sample/df_missing.csv", header=0, index_col=False)

print(df.describe())
def float_equal(a, b, epsilon=1e-9):
    return abs(a - b) < epsilon

# Filter the rows where the last column is missing (NaN)
missing_last_col = df[df.iloc[:, -1].isnull()]

print(missing_last_col)
# Count the occurrences of 1 and 0 in the first column
count_1 = missing_last_col[float_equal(missing_last_col.iloc[:, 1],1.0)].shape[0]
count_0 = missing_last_col[float_equal(missing_last_col.iloc[:, 1],0.0)].shape[0]

print(f"Number of items with 1 in the first column: {count_1}")
print(f"Number of items with 0 in the first column: {count_0}")
