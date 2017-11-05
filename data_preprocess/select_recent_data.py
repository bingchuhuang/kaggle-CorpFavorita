import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from datetime import date

df_train = pd.read_csv('../data/train.csv')
df_train['date'] = pd.to_datetime(df_train['date'])
sd = date(2017,2,15)
ed = date(2017,8,15)

sub_df_train = df_train.loc[(df_train['date'] > sd) & (df_train['date'] < ed)]

outname = 'train_' + sd.isoformat() + '_to_' + ed.isoformat() + '.csv'
sub_df_train.to_csv(outname,header=True,index=False,mode='w')
