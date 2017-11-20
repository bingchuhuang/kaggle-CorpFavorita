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
import numpy as np
sys.path.append('../')
from config import config
from os import path, makedirs
import time
from datetime import date

def merge_data(df, df_oil, df_holidays, df_items, df_stores):
    grouped = df.groupby('date')
    df_combine_list = []
    for _, group in grouped:
        #print('merging date  ',date)
        group = pd.merge(group, df_oil, on='date', how='left')
        group = pd.merge(group, df_holidays, on='date', how='left')
        group = pd.merge(group, df_items, on='item_nbr', how='left')
        group = pd.merge(group, df_stores, on='store_nbr', how='left')
        df_combine_list.append(group)

    return pd.concat(df_combine_list, axis=0)

def concat_data(df, df_sales_per_transaction, df_sales_per_holiday, df_sales_per_promo, df_sales_per_dow, df_sales_per_dom):
    grouped = df.groupby('store_nbr')
    df_combine_list = []
    for _, group in grouped:
        group = pd.merge(group, df_sales_per_transaction, on='store_nbr', how='left')
        group = pd.merge(group, df_sales_per_holiday, on=['store_nbr','type','description','locale','locale_name','transferred'], how='left')
        group = pd.merge(group, df_sales_per_promo, on=['store_nbr','onpromotion'], how='left')
        group = pd.merge(group, df_sales_per_dow, on=['store_nbr','dow'], how='left')
        group = pd.merge(group, df_sales_per_dom, on=['store_nbr','dom'], how='left')
        df_combine_list.append(group)
    
    return pd.concat(df_combine_list, axis=0)

def sales_per_transaction(df_train, df_transactions):
    df_sales_sum = df_train[['store_nbr','unit_sales','onpromotion']].groupby('store_nbr').sum()
    df_transactions_sum = df_transactions.groupby('store_nbr').sum()
    
    df_sales_transactions = pd.concat([df_sales_sum,df_transactions_sum], axis=1)
    df_sales_transactions['shopavg_transaction'] = df_sales_transactions['unit_sales']/df_sales_transactions['transactions']
    df_sales_transactions['store_nbr'] = df_sales_transactions.index
    print(df_sales_transactions.head(5))
    return df_sales_transactions[['store_nbr','shopavg_transaction']]

def sales_per_holiday(df):
    grouped = df.groupby(['store_nbr','type','description','locale','locale_name','transferred'])
    df_sales_holidays = grouped.apply(lambda x: x['unit_sales'].sum()/len(x)).reset_index(name='shopavg_holiday')
    print(df_sales_holidays.head(5))
    return df_sales_holidays


def sales_per_promo(df):
    grouped = df.groupby(['store_nbr','onpromotion'])
    df_sales_promo = grouped.apply(lambda x: x['unit_sales'].sum()/len(x)).reset_index(name='shopavg_promo')
    print(df_sales_promo.head(5))
    return df_sales_promo

def sales_per_weekday(df):
    grouped = df.groupby(['store_nbr','dow'])
    df_sales_dow = grouped.apply(lambda x: x['unit_sales'].sum()/len(x)).reset_index(name='shopavg_dow')
    print(df_sales_dow.head(5))
    return df_sales_dow

def sales_per_monthday(df):
    grouped = df.groupby(['store_nbr','dom'])
    df_sales_dom = grouped.apply(lambda x: x['unit_sales'].sum()/len(x)).reset_index(name='shopavg_dom')
    print(df_sales_dom.head(5))
    return df_sales_dom

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

dict_item        = {k: v for v,k in enumerate(df_items.item_nbr.unique()) }
dict_class       = {k: v for v,k in enumerate(df_items['class'].unique()) }
dict_family      = {k: v for v,k in enumerate(df_items.family.unique()) }
dict_city        = {k: v for v,k in enumerate(df_stores.city.unique()) }
dict_state       = {k: v for v,k in enumerate(df_stores.state.unique()) }
dict_stype       = {k: v for v,k in enumerate(np.sort(df_stores.stype.unique())) }
dict_type        = {k: v for v,k in enumerate(df_holidays.type.unique()) }
dict_description = {k: v for v,k in enumerate(df_holidays.description.unique()) }
dict_locale      = {k: v for v,k in enumerate(df_holidays.locale.unique()) }
dict_locale_name = {k: v for v,k in enumerate(df_holidays.locale_name.unique()) }

df_items =  df_items.replace({'item_nbr':dict_item})
df_items  = df_items.replace({'class':dict_class})
df_items  = df_items.replace({'family':dict_family})
df_stores = df_stores.replace({'city':dict_city})
df_stores = df_stores.replace({'state':dict_state})
df_stores = df_stores.replace({'stype':dict_stype})
df_holidays = df_holidays.replace({'type':dict_type})
df_holidays = df_holidays.replace({'description':dict_description})
df_holidays = df_holidays.replace({'locale':dict_locale})
df_holidays = df_holidays.replace({'locale_name':dict_locale_name})

print('data_preprocess: preprocessing data ...')

start_time = time.time()
df_train = merge_data(df_train, df_oil, df_holidays, df_items, df_stores)
df_test  = merge_data(df_test, df_oil, df_holidays, df_items, df_stores)
elapsed_time = time.time() - start_time
print('Time spent on merging: ', elapsed_time)
print('df_train:')
print(df_train.head(5))
print('df_test:')
print(df_test.head(5))

df_train = df_train.fillna(-1)
df_train['type']        = df_train['type'].astype(int)
df_train['description'] = df_train['description'].astype(int)
df_train['locale']      = df_train['locale'].astype(int)
df_train['locale_name'] = df_train['locale_name'].astype(int)
df_train['transferred'] = df_train['transferred'].astype(int)
df_train['onpromotion'] = df_train['onpromotion'].astype(int)

df_train['dow'] = pd.to_datetime(df_train['date']).dt.weekday
df_train['dom'] = pd.to_datetime(df_train['date']).dt.day
df_test['dow']  = pd.to_datetime(df_test['date']).dt.weekday
df_test['dom']  = pd.to_datetime(df_test['date']).dt.day

# sales related feature need to be handled here

# calculate shop average sales per transaction
df_sales_per_transaction = sales_per_transaction(df_train, df_transactions)

# calculate shop average sales per holiday
df_sales_per_holiday = sales_per_holiday(df_train)


# calculate shop average sales per promotion 
df_sales_per_promo = sales_per_promo(df_train)

# calculate shop average sales per weekday
df_sales_per_dow = sales_per_weekday(df_train)

# calculate shop average sales per monthday
df_sales_per_dom = sales_per_monthday(df_train)


# concatenate dataframes
df_train = concat_data(df_train, df_sales_per_transaction, df_sales_per_holiday, df_sales_per_promo, df_sales_per_dow, df_sales_per_dom)
df_test  = concat_data(df_test,  df_sales_per_transaction, df_sales_per_holiday, df_sales_per_promo, df_sales_per_dow, df_sales_per_dom)

print('data_preprocess: data precessed!') 
print('df_train:')
print(df_train.head(5))
print('df_test:')
print(df_test.head(5))


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
