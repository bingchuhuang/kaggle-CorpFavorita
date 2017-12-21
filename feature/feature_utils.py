'''
__file__
    feature_utils.py

__description__
    functions for computing features 

__author__
    Bingchu Huang

'''

import pandas as pd
from datetime import date, timedelta
import numpy as np
from scipy import stats
import sys
sys.path.append('../')
from config import config
import pickle
from os import path
import gc

# read directly
def simple_read(df, feature):
    return df[feature]

# date
def day_of_week(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.weekday #fastest way
    return df[feature]

def day_of_month(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.daysinmonth #fastest way
    return df[feature]

def day_of_year(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.dayofyear
    return df[feature]

def week(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.weak
    return df[feature]

def week_of_year(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.weekofyear
    return df[feature]

def month(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.month
    return df[feature]

def year(df, feature):
    df[feature] = pd.to_datetime(df['date']).dt.year
    return df[feature]

# store
def store_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_stores+'.pkl'),'rb') as f:
        df_stores = pickle.load(f)
    df_feature = pd.merge(df[['store_nbr']], df_stores, on='store_nbr', how='left')
    print(df_feature.info())
    print(df_feature.head(5))
    return df_feature[feature]

# item
def item_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_items+'.pkl'),'rb') as f:
        df_items = pickle.load(f)
    df_feature = pd.merge(df[['item_nbr']], df_items, on='item_nbr', how='left')
    return df_feature[feature]

# store x item
# to be added...

# holiday
def holiday_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_holiday_counts+'.pkl'),'rb') as f:
        df_holidays = pickle.load(f)
    if 'national' in feature:
        df_feature = pd.merge(df[['date']], df_holidays, on='date', how='left')
    else:
        df_feature = pd.merge(df[['date','store_nbr']], df_holidays, on='date', how='left')
    return df_feature[feature]

# weather
def weather_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_weather+'.pkl'),'rb') as f:
        df_weather = pickle.load(f)
    df_feature = pd.merge(df[['date','store_nbr']], df_weather, on=['date','store_nbr'], how='left')
    return df_feature[feature]

# transaction 
def transaction_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_transactions+'.pkl'),'rb') as f:
        df_transactions = pickle.load(f)
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    df['date'] = pd.to_datetime(df['date'])

    period = 'dow'
    if 'dow' in feature:
        df_transactions['dow'] = df_transactions['date'].dt.dayofweek.astype(np.int16)
        df['dow'] = df['date'].dt.dayofweek.astype(np.int16)
        period = 'dow'
    
    if 'quarter' in feature:
        df_transactions['quarter'] = df_transactions['date'].dt.quarter.astype(np.int16)
        df['quarter'] = df['date'].dt.quarter.astype(np.int16)
        period = 'quarter'
    
    if 'mon' in feature:
        df_transactions['mon'] = df_transactions['date'].dt.month.astype(np.int16)
        df['mon'] = df['date'].dt.month.astype(np.int16)
        period = 'mon'

    grouped  = df_transactions[['store_nbr',period,'transactions']].groupby(['store_nbr',period])
    if 'med' in feature:
        df_out = grouped['transactions'].median().reset_index()

    if 'mean' in feature:
        df_out = grouped['transactions'].mean().reset_index()

    if 'std' in feature:
        df_out = grouped['transactions'].std().reset_index()
    
    if 'skew' in feature:
        df_out = grouped['transactions'].apply(stats.skew).reset_index()
    
    if 'kurtosis' in feature:
        df_out = grouped['transactions'].apply(stats.kurtosis).reset_index()
    
    if 'hm' in feature:
        df_out = grouped['transactions'].apply(stats.hmean).reset_index()
    
    if '10pct' in feature:
        df_out = grouped['transactions'].quantile(0.1).reset_index()
    
    if '90pct' in feature:
        df_out = grouped['transactions'].quantile(0.9).reset_index()

    df_out.rename(columns={'transactions':feature},inplace=True)
    df_feature = pd.merge(df[['store_nbr',period]], df_out, on=['store_nbr',period], how='left')
    return df_feature[feature]
 
def sales_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train['year'] = df_train['date'].dt.year.astype(np.int16)
    df['year'] = df['date'].dt.year.astype(np.int16)
    period = 'year'
    if 'dow' in feature:
        df_train['dow'] = df_train['date'].dt.dayofweek.astype(np.int16)
        df['dow'] = df['date'].dt.dayofweek.astype(np.int16)
        period = 'dow'
    
    if 'quarter' in feature:
        df_train['quarter'] = pd.to_datetime(df_train['date']).dt.quarter.astype(np.int16)
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter.astype(np.int16)
        period = 'quarter'
    
    if 'mon' in feature:
        df_train['mon'] = pd.to_datetime(df_train['date']).dt.month.astype(np.int16)
        df['mon'] = pd.to_datetime(df['date']).dt.month.astype(np.int16)
        period = 'mon'

    grouped  = df_train[['store_nbr','item_nbr',period,'unit_sales']].groupby(['item_nbr','store_nbr',period])
    if 'med' in feature:
        df_out = grouped['unit_sales'].median().reset_index()

    if 'mean' in feature:
        df_out = grouped['unit_sales'].mean().reset_index()

    if 'std' in feature:
        df_out = grouped['unit_sales'].std().reset_index()
    
    if 'skew' in feature:
        df_out = grouped['unit_sales'].apply(stats.skew).reset_index()
    
    if 'kurtosis' in feature:
        df_out = grouped['unit_sales'].apply(stats.kurtosis).reset_index()
    
    if 'hm' in feature:
        df_out = grouped['unit_sales'].apply(stats.hmean).reset_index()
    
    if '10pct' in feature:
        #df_out = grouped['unit_sales'].quantile(0.1).reset_index() # very slow
        df_out = grouped['unit_sales'].apply(lambda x: np.percentile(x, q=10)).reset_index()
    
    if '90pct' in feature:
        #df_out = grouped['unit_sales'].quantile(0.9).reset_index()
        df_out = grouped['unit_sales'].apply(lambda x: np.percentile(x, q=90)).reset_index()
        
    df_out.rename(columns={'unit_sales':feature},inplace=True)
    df_feature = pd.merge(df[[period,'item_nbr','store_nbr']], df_out, on=[period,'item_nbr','store_nbr'], how='left')
    return df_feature[feature]

# moving average
def ma_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    df_train['date'] = pd.to_datetime(df_train['date'])
    with open(path.join(config.proc_data_path,config.fname_items+'.pkl'),'rb') as f:
        df_items = pickle.load(f)
    df_train = pd.merge(df_train, df_items, how='left', on=['item_nbr'])
    df = pd.merge(df, df_items, how='left', on=['item_nbr'])
    df_train['date'] = pd.to_datetime(df_train['date'])
    df['date'] = pd.to_datetime(df['date'])

    mlist = ['store_nbr']
    if 'ma_is' in feature:
        mlist.append('item_nbr')
    if 'ma_cs' in feature:
        mlist.append('class')
    if 'ma_fs' in feature:
        mlist.append('family')
        
    flist = mlist.copy()
    flist.append('unit_sales')
    
    if 'tot' in feature:
        ma = df_train[flist].groupby(mlist)['unit_sales'].mean().to_frame(feature).reset_index()
    else:
        ndays = feature.split('_')[-1]
        last_date = df_train.iloc[df_train.shape[0]-1].date
        print('ndays = ',ndays)
        print('last_date = ',last_date)
        print('date = ',last_date-timedelta(int(ndays)))
        tmp = df_train[df_train.date>last_date-timedelta(int(ndays))]
        print('tmp.shape = ',tmp.shape)
        print(tmp.head(5))
        print(tmp.info())
        ma = tmp[flist].groupby(mlist)['unit_sales'].mean().to_frame(feature).reset_index()

    df_feature = pd.merge(df, ma, on=mlist, how='left')
    return df_feature[feature]

def ma_wt(df, feature):
    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    df_train['date'] = pd.to_datetime(df_train['date'])
    with open(path.join(config.proc_data_path,config.fname_items+'.pkl'),'rb') as f:
        df_items = pickle.load(f)
    df_train = pd.merge(df_train, df_items, how='left', on=['item_nbr'])
    df = pd.merge(df, df_items, how='left', on=['item_nbr'])
    df_train['date'] = pd.to_datetime(df_train['date'])
    df['date'] = pd.to_datetime(df['date'])
    df_train['dow'] = df_train['date'].dt.dayofweek
    df['dow'] = df['date'].dt.dayofweek

    mlist = ['store_nbr']
    if 'ma_is' in feature:
        mlist.append('item_nbr')
    if 'ma_cs' in feature:
        mlist.append('class')
    if 'ma_fs' in feature:
        mlist.append('family')

    flist = mlist.copy()
    flist.append('unit_sales')

    ma = df_train[flist].groupby(mlist)['unit_sales'].mean().to_frame('ma')
    
    mlist_dw = mlist.copy()
    mlist_dw.append('dow') 
    flist_dw = flist.copy()
    flist_dw.append('dow')

    lastdate = df_train.iloc[df_train.shape[0]-1].date
    ma_dw = df_train[flist_dw].groupby(mlist_dw)['unit_sales'].mean().to_frame('ma_dw')
    ma_dw.reset_index(inplace=True)

    for i in [112,56,28,14,7,3,1]:
        tmp = df_train[df_train.date>lastdate-timedelta(int(i))]
        tmpg = tmp.groupby(mlist)['unit_sales'].mean().to_frame(feature+str(i))
        ma = ma.join(tmpg, how='left')
    del tmp, tmpg
    gc.collect()

    flist_wk = mlist_dw.copy()
    flist_wk.append('ma_dw')
    print(ma_dw.head(5))
    ma_wk = ma_dw[flist_wk].groupby(mlist)['ma_dw'].mean().to_frame('ma_wk')
    ma_wk.reset_index(inplace=True)
    ma['ma'] = ma.median(axis=1)
    ma.reset_index(inplace=True)

    df_feature = pd.merge(df,         ma,    how='left', on=mlist)
    df_feature = pd.merge(df_feature, ma_wk, how='left', on=mlist)
    df_feature = pd.merge(df_feature, ma_dw, how='left', on=mlist_dw)
    df_feature[feature] = df_feature.ma
    pos_idx = df_feature['ma_wk'] > 0
    df_pos = df_feature.loc[pos_idx]
    df_feature.loc[pos_idx, feature] = df_pos['ma'] * df_pos['ma_dw'] / df_pos['ma_wk']
    df_feature.loc[:,feature].fillna(0, inplace=True)

    return df_feature[feature]

def oil_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_oil+'.pkl'),'rb') as f:
        df_oil = pickle.load(f)
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil['dcoilwtico'].interpolate(inplace=True)
    df_feature = pd.merge(df[['date']], df_oil, on=['date'], how='left')
    print(df.info())
    print(df_oil.info())
    print(df_feature.info())
    print(df_feature.head(5))
    return df_feature[feature]

def item_sold_days(df, feature):
    df[feature] = pd.to_datetime(df['date']) - pd.to_datetime(df['item_start_date'])
    return df[feature]

def get_prev(df,period_index,prevperiod_index,name):
    df_prev = df[[period_index]]
    df_prev.columns.rename(columns={period_index:prevperiod_index,'unite_sales':name}, inplace=True)
    return df_prev

