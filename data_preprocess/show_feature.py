import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

df_train = pd.read_csv('train_2017-06-15_to_2017-08-15.csv')
#df_test = pd.read_csv('../data/test.csv')
#df_sample = pd.read_csv('../data/sample_submission.csv')
df_stores = pd.read_csv('../data/stores.csv')
df_items = pd.read_csv('../data/items.csv')
df_transactions = pd.read_csv('../data/transactions.csv')
df_oil = pd.read_csv('../data/oil.csv')
df_holidays_events = pd.read_csv('../data/holidays_events.csv')

df_stores['stype'] = df_stores['type']
df_stores = df_stores.drop(['type'],axis=1)

dict_family = {k: v for v,k in enumerate(df_items.family.unique()) }
dict_city = {k: v for v,k in enumerate(df_stores.city.unique()) }
dict_state = {k: v for v,k in enumerate(df_stores.state.unique()) }
dict_stype = {k: v for v,k in enumerate(np.sort(df_stores.stype.unique())) }
dict_type = {k: v for v,k in enumerate(df_holidays_events.type.unique()) }
dict_holidays = {k: v for v,k in enumerate(df_holidays_events.description.unique()) }
dict_locale = {k: v for v,k in enumerate(df_holidays_events.locale.unique()) }
dict_locale_name = {k: v for v,k in enumerate(df_holidays_events.locale_name.unique()) }

print('-------- train --------')
print(df_train.head(5))


print('-------- stores --------')
print(df_stores.head(5))

print('-------- items --------')
print(df_items.head(5))

print('-------- transactions --------')
print(df_transactions.head(5))

print('-------- oil --------')
print(df_oil.head(5))

print('-------- holidays and events --------')
print(df_holidays_events.head(5))

# map names to numbers by dictionaries
df_items = df_items.replace({'family':dict_family})
df_stores = df_stores.replace({'city':dict_city, 'state':dict_state, 'stype':dict_stype})
df_holidays_events = df_holidays_events.replace({'type':dict_type,'locale':dict_locale,'locale_name':dict_locale_name,'description':dict_holidays})

start_time = time.time()

grouped = df_train.groupby('date')
df_combine_list = []
for date, group in grouped:
    #print('merging date  ',date)
    group = pd.merge(group, df_transactions, on=['date','store_nbr'], how='left')
    group = pd.merge(group, df_oil, on='date', how='left')
    group = pd.merge(group, df_holidays_events, on='date', how='left')
    group = pd.merge(group, df_items, on='item_nbr', how='left')
    group = pd.merge(group, df_stores, on='store_nbr', how='left')
    df_combine_list.append(group)

df_combine = pd.concat(df_combine_list, axis=0)
elapsed_time = time.time() - start_time
print('Used time on merging: ', elapsed_time)

print('-------- combined data --------')
print(df_combine.head(10))

outname='combine_data_2017-06-15_to_2017-08-15.csv'
df_combine.to_csv(outname,header=True,index=False,mode='w')
