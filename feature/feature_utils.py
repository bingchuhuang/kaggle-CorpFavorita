'''
__file__
    feature_utils.py

__description__
    functions for computing features 

__author__
    Bingchu Huang

'''

import pandas as pd
from datetime import date
import numpy as np
from scipy import stats
import sys
sys.path.append('../')
from config import config
import pickle
from os import path

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
    df_transactions['year'] = df_transactions['date'].dt.year.astype(np.int16)
    df_transactions['mon'] = df_transactions['date'].dt.month.astype(np.int8)
    df['year'] = pd.to_datetime(df['date']).dt.year.astype(np.int16)
    df['mon'] = pd.to_datetime(df['date']).dt.month.astype(np.int8)

    grouped  = df_transactions.groupby(['year','mon','store_nbr'])
    if 'med' in feature:
        df_out   = grouped['transactions'].median().reset_index()

    if 'mean' in feature:
        df_out = grouped['transactions'].mean().reset_index()

    if 'std' in feature:
        df_out= grouped['transactions'].std().reset_index()
    
    if 'skew' in feature:
        df_out= grouped['transactions'].apply(stats.skew).reset_index()
    
    if 'kurtosis' in feature:
        df_out= grouped['transactions'].apply(stats.kurtosis).reset_index()
    
    if 'hm' in feature:
        df_out= grouped['transactions'].apply(stats.hmean).reset_index()
    
    if '10pct' in feature:
        df_out= grouped['transactions'].quantile(0.1).reset_index()
    
    if '90pct' in feature:
        df_out= grouped['transactions'].quantile(0.9).reset_index()

    df_out.rename(columns={'transactions':feature},inplace=True)
    df_feature = pd.merge(df[['year','mon','store_nbr']], df_out, on=['year','mon','store_nbr'], how='left')
    return df_feature[feature]
 
def sales_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train['year'] = df_train['date'].dt.year.astype(np.int16)
    df_train['mon'] = df_train['date'].dt.month.astype(np.int8)
    
    df['year'] = pd.to_datetime(df['date']).dt.year.astype(np.int16)
    df['mon'] = pd.to_datetime(df['date']).dt.month.astype(np.int8)

    grouped  = df_train.groupby(['year','mon','item_nbr'])
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
        df_out= grouped['unit_sales'].quantile(0.1).reset_index()
    
    if '90pct' in feature:
        df_out= grouped['unit_sales'].quantile(0.9).reset_index()

    df_out.rename(columns={'unit_sales':feature},inplace=True)
    df_feature = pd.merge(df[['year','mon','item_nbr']], df_out, on=['year','mon','item_nbr'], how='left')
    return df_feature[feature]

def oil_feature(df, feature):
    with open(path.join(config.proc_data_path,config.fname_oil+'.pkl'),'rb') as f:
        df_oil = pickle.load(f)
    df_feature = pd.merge(df[['date']], df_oil, on=['date'], how='left')
    return df_feature[feature]


def item_sold_days(df, feature):
    df[feature] = pd.to_datetime(df['date']) - pd.to_datetime(df['item_start_date'])
    return df[feature]

def get_prev(df,period_index,prevperiod_index,name):
    df_prev = df[[period_index]]
    df_prev.columns.rename(columns={period_index:prevperiod_index,'unite_sales':name}, inplace=True)
    return df_prev

def prevperiod(df_test, df, feature):
    period = 'quarter'
    if 'month' in feature:
        period = 'month'
    if 'halfyear' in feature:
        period = 'halfyear'
    if 'year' in feature:
        period = 'year'
    
    grp_tag = feature.split('_')[-1]
    print('grp_tag = ',grp_tag)

    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    df_test['quarter'] = pd.to_datetime(df_test['date']).dt.quarter
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df_test['year'] = pd.to_datetime(df_test['date']).dt.year
    if 'month' not in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.month
        df_test['month'] = pd.to_datetime(df_test['date']).dt.month
    if 'day' not in df.columns:
        df['day'] = pd.to_datetime(df['date']).dt.day
        df_test['day'] = pd.to_datetime(df_test['date']).dt.day

    period_index = period + '_index'
    prevperiod_index = 'prev' + period + '_index'

    if period=='month':
        df[period_index] = df['year']*12 + df['month']-1
        df_test[period_index] = df_test['year']*12 + df_test['month']-1

    if period=='quarter':
        df[period_index] = df['year']*4 + df['quarter']-1
        df_test[period_index] = df_test['year']*4 + df_test['quarter']-1

    if period=='halfyear':
        df[period_index] = df['year']*2 + np.floor((df['quarter']-1)/2)
        df_test[period_index] = df_test['year']*2 + np.floor((df_test['quarter']-1)/2)

    if period=='year':
        df[period_index] = df['year']
        df_test[period_index] = df_test['year']

    df[prevperiod_index] = df[period_index] - 1
    df_test[prevperiod_index] = df_test[period_index] - 1

    grp_list = []
    if 'd' in grp_tag:
        grp_list.append('day')
    if 'p' in grp_tag:
        grp_list.append('onpromotion')
    if 'h' in grp_tag:
        grp_list.append('locale')
    if 's' in grp_tag:
        grp_list.append('store_nbr')

    grouped = df.groupby(grp_list)
    df_med   = grouped['unit_sales'].median().reset_index()
    df_mean  = grouped['unit_sales'].mean().reset_index()
    df_std   = grouped['unit_sales'].std().reset_index()
    df_skew  = grouped['unit_sales'].apply(stats.skew).reset_index()
    df_kurt  = grouped['unit_sales'].apply(stats.kurtosis).reset_index()
    df_hmean = grouped['unit_sales'].apply(stats.hmean).reset_index()
    df_10pct = grouped['unit_sales'].quantile(0.1).reset_index()
    df_90pct = grouped['unit_sales'].quartile(0.9).reset_index()

    name_med = 'prev_' + period + '_' + grp_tag + '_med'
    df_prev_med = get_prev(df_med,period_index,prevperiod_index,name_med)    

    name_mean = 'prev_' + period + '_' + grp_tag + '_mean'
    df_prev_mean = get_prev(df_mean,period_index,prevperiod_index,name_mean)    
    
    name_std = 'prev_' + period + '_' + grp_tag + '_std'
    df_prev_std = get_prev(df_std,period_index,prevperiod_index,name_std)    

    name_skew = 'prev_' + period + '_' + grp_tag + '_skew'
    df_prev_skew = get_prev(df_skew,period_index,prevperiod_index,name_skew)    
    
    name_kurt = 'prev_' + period + '_' + grp_tag + '_kurt'
    df_prev_kurt = get_prev(df_kurt,period_index,prevperiod_index,name_kurt)    

    name_hmean = 'prev_' + period + '_' + grp_tag + '_hmean'
    df_prev_hmean = get_prev(df_hmean,period_index,prevperiod_index,name_hmean)

    name_10pct = 'prev_' + period + '_' + grp_tag + '_10pct'
    df_prev_10pct = get_prev(df_10pct,period_index,prevperiod_index,name_10pct)    
    
    name_90pct = 'prev_' + period + '_' + grp_tag + '_90pct'
    df_prev_90pct = get_prev(df_90pct,period_index,prevperiod_index,name_90pct)    
    

    df_ret = pd.merge(df_test,df_prev_med,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_mean,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_std,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_skew,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_kurt,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_hmean,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_10pct,on=[prevperiod_index],how='left')
    df_ret = pd.merge(df_ret,df_prev_90pct,on=[prevperiod_index],how='left')

    return df_ret
