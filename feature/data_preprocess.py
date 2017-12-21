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
#import json
from datetime import date, datetime, timedelta
from scipy import stats

def dump_df(df,path_name):
    out = pickle.dumps(df)
    n_bytes = len(out)
    max_bytes = 2**31 - 1
    with open(path_name,'wb') as f:
        for i in range(0, n_bytes, max_bytes):
            f.write(out[i:i+max_bytes])

def count_day_before(df_holidays, df_holiday_counts, htype):
    df_holidays = df_holidays[(df_holidays['locale']==htype) & (df_holidays['transferred']==False)]
    htype_name = 'day_before_holiday'
    locale_name = 'locale_name'
    if htype==0:
        htype_name += '_local'
        locale_name += '_local'
    if htype==1:
        htype_name += '_regional'
        locale_name += '_regional'
    if htype==2:
        htype_name += '_national'

    df_holiday_counts[htype_name] = np.zeros(df_holiday_counts.shape[0],dtype=np.int16)
    if htype != 2:
        df_holiday_counts[locale_name] = -1
    index_holiday = 0
    next_holiday = df_holidays.iloc[index_holiday]
    for i, row in df_holiday_counts.iterrows():
        day = row['date']
        if day < next_holiday['date']:
            df_holiday_counts.at[df_holiday_counts.index[i],htype_name] = (next_holiday['date'] - day).days
            if htype != 2:
                df_holiday_counts.at[df_holiday_counts.index[i],locale_name] = next_holiday['locale']
        else:
            while day > next_holiday['date']:
                index_holiday += 1
                next_holiday = df_holidays.iloc[index_holiday]
    return df_holiday_counts

def count_day_after(df_holidays, df_holiday_counts, htype):

    df_holidays = df_holidays[(df_holidays['locale']==htype) & (df_holidays['transferred']==False)]

    htype_name = 'day_after_holiday'
    locale_name = 'locale_name'
    if htype==0:
        htype_name += '_local'
        locale_name += '_local'
    if htype==1:
        htype_name += '_regional'
        locale_name += '_regional'
    if htype==2:
        htype_name += '_national'

    df_holiday_counts[htype_name] = np.zeros(df_holiday_counts.shape[0],dtype=np.int16)

    if htype != 2:
        df_holiday_counts[locale_name] = -1
    index_holiday = 0
    prev_holiday = df_holidays.iloc[index_holiday]
    next_holiday = df_holidays.iloc[index_holiday+1]
    for i, row in df_holiday_counts.iterrows():
        day = row['date']
        if day < next_holiday['date'] and day > prev_holiday['date']:
            df_holiday_counts.at[df_holiday_counts.index[i],htype_name] = (day - prev_holiday['date']).days
            if htype != 2:
                df_holiday_counts.at[df_holiday_counts.index[i],locale_name] = prev_holiday['locale']
        else:
            while day > next_holiday['date']:
                index_holiday += 1
                prev_holiday = df_holidays.iloc[index_holiday]
                next_holiday = df_holidays.iloc[index_holiday+1]
    return df_holiday_counts

def count_holidays(df_holidays):
    df_holidays['isHoliday'] = np.zeros(df_holidays.shape[0],dtype=np.int8)
    df_holidays.loc[df_holidays['transferred']==False, 'isHoliday'] = 1

    df_holiday_perweek = df_holidays[['year','woy','isHoliday']].groupby(['year','woy']).sum().reset_index()
    df_holiday_perweek.rename(columns={'isHoliday':'holidays_thisweek'}, inplace=True)
    df_holiday_perweek['holidays_lastweek'] = df_holiday_perweek['holidays_thisweek'].shift(1)
    df_holiday_perweek['holidays_nextweek'] = df_holiday_perweek['holidays_thisweek'].shift(-1)
    df_holiday_perweek = df_holiday_perweek.fillna(0)
    df_holiday_perweek['year'] = df_holiday_perweek['year'].astype(np.int16)
    df_holiday_perweek['woy'] = df_holiday_perweek['woy'].astype(np.int8)
    df_holiday_perweek['holidays_thisweek'] = df_holiday_perweek['holidays_thisweek'].astype(np.int8)
    df_holiday_perweek['holidays_lastweek'] = df_holiday_perweek['holidays_lastweek'].astype(np.int8)
    df_holiday_perweek['holidays_nextweek'] = df_holiday_perweek['holidays_nextweek'].astype(np.int8)
    
    df_holidays = pd.merge(df_holidays,df_holiday_perweek,on=['year','woy'],how='left')
    return df_holidays

def process_train():

    print('processing train data...')
    dict_dtype_train={'id':np.int32,'date':object,'store_nbr':np.int8,'item_nbr':np.int32,'unit_sales':np.float32,'onpromotion':bool}
    reader = pd.read_csv(path.join(config.data_path,config.fname_train+'.csv'), dtype=dict_dtype_train, parse_dates=['date'],
                         usecols=range(0,6), skiprows=range(1,107758057), iterator=True) #skip date before 2017-03-01
    try:
        df_train = reader.get_chunk(1e8)
    except StopIteration:
        print('Read train iteration is stopped.')
   
    # creating records for all items, in all markets on all dates #from Paulo Pinto:
    # https://www.kaggle.com/paulorzp/log-means-and-medians-to-predict-new-itens-0-546/code
    # for correct calculation of daily unit sales averages.
    u_dates = df_train.date.unique()
    u_stores = df_train.store_nbr.unique()
    u_items = df_train.item_nbr.unique()
    df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
    df_train = df_train.reindex(
            pd.MultiIndex.from_product(
                        (u_dates, u_stores, u_items),
                        names=["date", "store_nbr", "item_nbr"]
                    )
    )

    del u_dates, u_stores, u_items

    df_train.fillna(0,inplace=True)
    df_train.reset_index(inplace=True)
    df_train.loc[(df_train.unit_sales<0),'unit_sales'] = 0 # remove return sales 
    df_train['unit_sales'] =  df_train['unit_sales'].apply(pd.np.log1p) #log(1+x)
    config.set_dtype(df_train)
    print(df_train.info())
    
    path_name = path.join(config.proc_data_path,config.fname_train+'.pkl')
    dump_df(df_train,path_name)
    print('process_train: Done!')

def process_test():

    print('processing test data...')
    dict_dtype_test={'id':np.int32,'date':object,'store_nbr':np.int8,'item_nbr':np.int32,'onpromotion':bool}
    reader_test = pd.read_csv(path.join(config.data_path,config.fname_test+'.csv'),  dtype=dict_dtype_test, parse_dates=['date'], usecols=range(0,5), iterator=True)
    try:
        df_test = reader_test.get_chunk(1e8)
    except StopIteration:
        print('Read test iteration is stopped.')
 
    df_test = df_test.fillna(0)
    config.set_dtype(df_test)
    print(df_test.info())

    path_name = path.join(config.proc_data_path,config.fname_test+'.pkl')
    dump_df(df_test,path_name)
    print('process_test: Done!')

def process_info():

    print('process_info: loading data ...')
    
    start_time = time.time()
    df_stores       = pd.read_csv(path.join(config.data_path,config.fname_stores+'.csv'),dtype={'store_nbr':np.int8,'city':np.string_,'state':np.string_,'type':np.string_,'cluster':np.int8})
    df_transactions = pd.read_csv(path.join(config.data_path,config.fname_transactions+'.csv'),dtype={'date':object,'store_nbr':np.int8,'transactions':np.int32})
    df_items        = pd.read_csv(path.join(config.data_path,config.fname_items+'.csv'),dtype={'item_nbr':np.int32,'family':np.string_,'class':np.int32,'perishable':np.int8})
    df_oil          = pd.read_csv(path.join(config.data_path,config.fname_oil+'.csv'),dtype={'date':object,'dcoilwtico':np.float16})
    df_holidays     = pd.read_csv(path.join(config.data_path,config.fname_holidays+'.csv'),dtype={'date':object,'type':np.string_,'locale':np.string_,'locale_name':np.string_,'description':np.string_,'transferred':bool})
    df_weather      = pd.read_csv(path.join(config.data_path,config.fname_weather+'.csv'),dtype={'date':object,'city':np.string_,'humidity':np.float16,'temperatureMin':np.float16,'temperatureMax':np.float16})
    
    df_stores['stype'] = df_stores['type']
    df_stores = df_stores.drop(['type'],axis=1)
 
    #dict_item        = {k: v for v,k in enumerate(df_items.item_nbr.unique()) }
    dict_class       = {k: v for v,k in enumerate(df_items['class'].unique()) }
    dict_family      = {k: v for v,k in enumerate(df_items.family.unique()) }
    dict_city        = {k: v for v,k in enumerate(df_stores.city.unique()) }
    dict_state       = {k: v for v,k in enumerate(df_stores.state.unique()) }
    dict_stype       = {k: v for v,k in enumerate(np.sort(df_stores.stype.unique())) }
    dict_type        = {k: v for v,k in enumerate(df_holidays.type.unique()) }
    dict_description = {k: v for v,k in enumerate(df_holidays.description.unique()) }
    dict_locale      = {k: v for v,k in enumerate(df_holidays.locale.unique()) }
    dict_locale_name = {k: v for v,k in enumerate(df_holidays.locale_name.unique()) }
    
    #df_items.replace({'item_nbr':dict_item}, inplace=True)
    print(df_items.head(5))
    df_items.replace({'class':dict_class}, inplace=True)
    df_items.replace({'family':dict_family}, inplace=True)
    df_stores.replace({'city':dict_city}, inplace=True)
    df_stores.replace({'state':dict_state}, inplace=True)
    df_stores.replace({'stype':dict_stype}, inplace=True)
    df_holidays.replace({'type':dict_type}, inplace=True)
    df_holidays.replace({'description':dict_description}, inplace=True)
    df_holidays.replace({'locale':dict_locale}, inplace=True)
    df_holidays.replace({'locale_name':dict_locale_name}, inplace=True)
    df_weather.replace({'city':dict_city}, inplace=True)

    df_weather = pd.merge(df_weather, df_stores, on='city', how='left')
    
    config.set_dtype(df_oil)
    config.set_dtype(df_items)
    config.set_dtype(df_stores)
    config.set_dtype(df_transactions)
    config.set_dtype(df_holidays)
    config.set_dtype(df_weather)
    
    print(df_oil.info())
    print(df_items.info())
    print(df_stores.info())
    print(df_transactions.info())
    print(df_holidays.info())
    print(df_weather.info())
    
    sd = date(2012,12,1)
    ed = date(2017,10,1)
    days = pd.date_range(sd, ed, freq='D')
    df_holiday_counts = pd.DataFrame({'date':days})
    df_holiday_counts['year'] = df_holiday_counts['date'].dt.year.astype(np.int16)
    df_holiday_counts['doy']  = df_holiday_counts['date'].dt.dayofyear.astype(np.int16)
    df_holiday_counts['woy']  = df_holiday_counts['date'].dt.week.astype(np.int8)

    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    #df_holidays['year'] = df_holidays['date'].dt.year.astype(np.int16)
    #df_holidays['doy']  = df_holidays['date'].dt.dayofyear.astype(np.int16)
    #df_holidays['woy']  = df_holidays['date'].dt.week.astype(np.int8)

    df_holiday_counts = pd.merge(df_holiday_counts, df_holidays, on=['date'], how='left') 
    df_holiday_counts = count_day_before(df_holidays, df_holiday_counts, 0)
    df_holiday_counts = count_day_before(df_holidays, df_holiday_counts, 1)
    df_holiday_counts = count_day_before(df_holidays, df_holiday_counts, 2)
    df_holiday_counts = count_day_after(df_holidays, df_holiday_counts, 0)
    df_holiday_counts = count_day_after(df_holidays, df_holiday_counts, 1)
    df_holiday_counts = count_day_after(df_holidays, df_holiday_counts, 2)
    df_holiday_counts = count_holidays(df_holiday_counts)

    df_holiday_counts.rename(columns={'locale_name_local':'city','locale_name_regional':'state'},inplace=True) 
    print(df_holiday_counts.info())
    print(df_stores.info())
    df_holiday_counts = pd.merge(df_holiday_counts, df_stores, on=['city','state'], how='left')
    df_holiday_counts = df_holiday_counts.fillna(0)

    config.set_dtype(df_holiday_counts)
    config.set_dtype(df_holidays)

    print('df_holiday_counts:')
    #print(df_holiday_counts.head(50))
    print(df_holiday_counts.info())
    
    print('df_holidays:')
    print(df_holidays.info())

    #df_train.replace({'item_nbr':dict_item}, inplace=True)
    #df_test.replace({'item_nbr':dict_item}, inplace=True)
    
    # dump maps
    
    #dicts = [dict_item, dict_class, dict_family, dict_city, dict_state, dict_stype, dict_type, dict_description, dict_locale, dict_locale_name]
    dicts = [dict_class, dict_family, dict_city, dict_state, dict_stype, dict_type, dict_description, dict_locale, dict_locale_name]
    
    with open('dict_maps.txt','w') as f:
        #f.write(str(dict_item)+'\n')
        f.write(str(dict_class)+'\n')
        f.write(str(dict_family)+'\n')
        f.write(str(dict_city)+'\n')
        f.write(str(dict_state)+'\n')
        f.write(str(dict_stype)+'\n')
        f.write(str(dict_type)+'\n')
        f.write(str(dict_description)+'\n')
        f.write(str(dict_locale)+'\n')
        f.write(str(dict_locale_name)+'\n')
        f.flush()
        f.close()
    
    print('data_preprocess: preprocessing data ...')
    
    
    '''
    +++++++++++++++++++++++++++++++++++++++++
    ++++++++++++++ save data ++++++++++++++++
    +++++++++++++++++++++++++++++++++++++++++
    '''
    
    start_time = time.time()
    proc_path = path.join(config.proc_data_path)
    if not path.exists(proc_path):
        makedirs(proc_path)
    
    path_name = path.join(config.proc_data_path,'dicts.pkl')
    dump_df(dicts,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_oil+'.pkl')
    dump_df(df_oil,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_items+'.pkl')
    dump_df(df_items,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_stores+'.pkl')
    dump_df(df_stores,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_transactions+'.pkl')
    dump_df(df_transactions,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_holidays+'.pkl')
    dump_df(df_holidays,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_holiday_counts+'.pkl')
    dump_df(df_holiday_counts,path_name)
    
    path_name = path.join(config.proc_data_path,config.fname_weather+'.pkl')
    dump_df(df_weather,path_name)
    
    print('process_info: Done!')
    
    elapsed_time = time.time() - start_time
    print('Time spent on saving data: ', elapsed_time)

def main():
    
    process_train()
    process_test()
    process_info()
   
if __name__ == '__main__':
    main()
