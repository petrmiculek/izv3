"""
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt

# %%

# load dataset
df = pd.read_pickle('accidents.pkl.gz')[['p11', 'p13a', 'p13b']]  # 487161 rows

# %%

# ignore drug-related crashes
df = df[df['p11'] != 4]
df = df[df['p11'] != 5]

# %%

## Heavy consequences

# append column with a flag indicating heavy crash consequences
df['conseq'] = False
df.loc[(df['p13a'] > 0) | (df['p13b'] > 0), 'conseq'] = True

print(df[df['conseq'] == True])  # 11785 rows

# %%

## heavy alcohol influence

# append column with a flag indicating heavy alcohol influence
df['alcohol'] = False
df.loc[(df['p11'] >= 7), 'alcohol'] = True

print(df[df['alcohol'] == True])  # 17378 rows

# %%

ct = pd.crosstab(df['alcohol'], df['conseq'])
print(ct)

# %%

out = scipy.stats.chi2_contingency(ct)
print(out[1])

"""
