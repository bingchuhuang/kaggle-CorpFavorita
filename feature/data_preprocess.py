'''
__file__
    data_preprocess.py

__description__
    preprocess data

__author__
    Bingchu Huang

'''

import sys
import pandas as pd
import pickle
sys.path.append('../')
from config import config
from os import path, makedirs
import time

def merge_data(df, df_transactions, df_oil, df_holidays, df_items, df_stores):
    grouped = df.groupby('date')
    df_combine_list = []
    for date, group in grouped:
        #print('merging date  ',date)
        group = pd.merge(group, df_transactions, on=['date','store_nbr'], how='left')
        group = pd.merge(group, df_oil, on='date', how='left')
        group = pd.merge(group, df_holidays, on='date', how='left')
        group = pd.merge(group, df_items, on='item_nbr', how='left')
        group = pd.merge(group, df_stores, on='store_nbr', how='left')
        df_combine_list.append(group)

    df = pd.concat(df_combine_list, axis=0)

'''
 load data
'''

print('data_preprocess: loading data ...')

df_train        = pd.read_csv(path.join(config.data_path,config.fname_train+'.csv')).fillna('')
df_test         = pd.read_csv(path.join(config.data_path,config.fname_test+'.csv')).fillna('')
df_stores       = pd.read_csv(path.join(config.data_path,config.fname_stores+'.csv'))
df_transactions = pd.read_csv(path.join(config.data_path,config.fname_transactions+'.csv'))
df_items        = pd.read_csv(path.join(config.data_path,config.fname_items+'.csv'))
df_oil          = pd.read_csv(path.join(config.data_path,config.fname_oil+'.csv'))
df_holidays     = pd.read_csv(path.join(config.data_path,config.fname_holidays+'.csv'))

df_stores['stype'] = df_stores['type']
df_stores = df_stores.drop(['type'],axis=1)

n_train, n_test = df_train.shape[0], df_test.shape[0]

print('data_preprocess: data loaded!')

'''
preprocess data
'''

print('data_preprocess: preprocessing data ...')


dict_family = {k: v for v,k in enumerate(df_items.family.unique()) }

df_items = df_items.replace({'family':dict_family})


start_time = time.time()
merge_data(df_train, df_transactions, df_oil, df_holidays, df_items, df_stores)
merge_data(df_test, df_transactions, df_oil, df_holidays, df_items, df_stores)
elapsed_time = time.time() - start_time
print('Time spent on merging: ', elapsed_time)
print('df_train:')
print(df_train.head(5))
print('df_test:')
print(df_test.head(5))

print('data_preprocess: data precessed!') 

'''
save data
'''

proc_path = path.join(config.proc_data_path)
if not path.exists(proc_path):
    makedirs(proc_path)

with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'wb') as f:
    pickle.dump(df_train, f, -1)

with open(path.join(config.proc_data_path,config.fname_test+'.pkl'),'wb') as f:
    pickle.dump(df_test, f, -1)

print('data_preprocess: Mission Accomplished!')
