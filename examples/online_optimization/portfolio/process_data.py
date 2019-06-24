import numpy as np
import pandas as pd
import os


DATA_FOLDER = '/home/gridsan/stellato/results/online/portfolio/'


'''
General
'''
# List all files with general
dfs = []
for f in os.listdir(DATA_FOLDER):
    if f.endswith(".csv") and 'general' in f:
        K = [int(s) for s in f.split("_") if s.isdigit()][0]

        # Read file and append horizon length
        df_K = pd.read_csv(os.path.join(DATA_FOLDER, f),
                           index_col=0, squeeze=True)
        df_K['K'] = K
        dfs.append(df_K)
df = pd.concat(dfs, axis=1).T.sort_values(by=['K'])

# Export updted table in csv
df.to_csv(os.path.join(DATA_FOLDER, 'complete.csv'))


'''
Detail
'''
# List all files with general
dfs_detail = []
for f in os.listdir(DATA_FOLDER):
    if f.endswith(".csv") and 'detail' in f:
        K = [int(s) for s in f.split("_") if s.isdigit()][0]

        # Read file and append horizon length
        df_K = pd.read_csv(os.path.join(DATA_FOLDER, f),
                           index_col=0)
        df_K['K'] = [K] * len(df_K)
        df_K.to_csv(os.path.join(DATA_FOLDER, '%d_full.csv' % K))
        dfs_detail.append(df_K)
df_detail = pd.concat(dfs_detail,
                      ignore_index=True).sort_values(by=['K']).reset_index(drop=True)

# Export updted table in csv
df_detail.to_csv(os.path.join(DATA_FOLDER, 'complete_full.csv'))
