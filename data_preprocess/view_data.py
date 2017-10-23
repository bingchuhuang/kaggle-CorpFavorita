import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
#import missingno as msno

#from subprocess import check_output

df_train = pd.read_csv('../data/train.csv')
#df_test = pd.read_csv('../data/test.csv')
#df_sample = pd.read_csv('../data/sample_submission.csv')
df_stores = pd.read_csv('../data/stores.csv')
df_items = pd.read_csv('../data/items.csv')
df_transactions = pd.read_csv('../data/transactions.csv')
df_oil = pd.read_csv('../data/oil.csv')
df_holidays_events = pd.read_csv('../data/holidays_events.csv')

#--- for integer type columns ---
def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

#--- for float type columns ---
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)            


# stores.csv
print('------- stores.csv -------')

print(df_stores.shape)
print(df_stores.head())

# Checking Missing Values
print('Any missing values: ', df_stores.isnull().values.any())

pp = pd.value_counts(df_stores.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_city.png', format='png')


print(df_stores.dtypes.unique())
print(df_stores.dtypes.nunique())

print(df_stores.store_nbr.nunique())


#--- Various cities distribution ---
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
ax = sns.countplot(y=df_stores['city'], data=df_stores) 

plt.savefig('fig/city.png', format='png')

#--- Various states distribution ---
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
ax = sns.countplot(y=df_stores['state'], data=df_stores) 
fig.tight_layout()
plt.savefig('fig/states.png', format='png')


#--- Various types ---
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax = sns.countplot(x="type", data=df_stores, palette="Set3")
fig.tight_layout()
plt.savefig('fig/types.png', format='png')


# Various types of stores distributed across different cities

fig, ax = plt.subplots()
fig.set_size_inches(16, 10)
ax = sns.countplot(x="city", hue="type", data=df_stores)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)

fig.tight_layout()
plt.savefig('fig/types_cities.png', format='png')

print(df_stores.cluster.sum())

# Distribution of all stores across cities

mm = (df_stores.groupby(['city']).sum())

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = mm.index, y= "cluster", data = mm)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
fig.tight_layout()
plt.savefig('fig/types_cities.png', format='png')

# 

obj_cols = list(df_stores.select_dtypes(include=['object']).columns)
for col in obj_cols:
    df_stores[col], _ = pd.factorize(df_stores[col])
    
print(df_stores.head(10))

# Memory check

mem = df_stores.memory_usage(index=True).sum()
print("Memory consumed by stores dataframe initially  :   {} MB" .format(mem/ 1024**2))

change_datatype(df_stores)
change_datatype_float(df_stores)

mem = df_stores.memory_usage(index=True).sum()
print("\n Memory consumed by stores dataframe later  :   {} MB" .format(mem/ 1024**2))

# items.csv
print('------- items.csv -------')
print(df_items.shape)
print(df_items.head())
print('Any missing values: ' , df_items.isnull().values.any())


pp = pd.value_counts(df_items.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_items.png', format='png')


print(df_items.dtypes.unique())
print(df_items.dtypes.nunique())


print(df_items.item_nbr.nunique())

# Distribution of various families of items

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.countplot(y = "family", data = df_items)
fig.tight_layout()
plt.savefig('fig/items_falimies.png', format='png')

# Distribution of perishable goods by family

mc = (df_items.groupby(['family']).sum())
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = mc.index, y= "perishable", data = mc)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 9)
fig.tight_layout()
plt.savefig('fig/goods_falimies.png', format='png')

# Distrbution of number of unique classes per family of items.

xc = df_items.groupby(['family'])['class'].nunique()
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
xc.plot.bar()
fig.tight_layout()
#plt.show()
plt.savefig('fig/unique_classes_family.png', format='png')

obj_cols = list(df_items.select_dtypes(include=['object']).columns)
for col in obj_cols:
    df_items[col], _ = pd.factorize(df_items[col])
    
print(df_items.head(10))

# Memory check

mem = df_items.memory_usage(index=True).sum()
print("Memory consumed by items dataframe initially  :   {} MB" .format(mem/ 1024**2))

change_datatype(df_items)
change_datatype_float(df_items)

mem = df_items.memory_usage(index=True).sum()
print("\n Memory consumed by items dataframe later  :   {} MB" .format(mem/ 1024**2))


# transactions.csv
print('--------- transactions.csv ---------')
print(df_transactions.shape)
print(df_transactions.head())

print('Any missing values: ', df_transactions.isnull().values.any())

pp = pd.value_counts(df_transactions.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_transactions.png', format='png')

print(df_transactions.dtypes.unique())
print(df_transactions.dtypes.nunique())

# Distribution of toal number of transactions made per individual store

pc = (df_transactions.groupby(['store_nbr']).sum())
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = pc.index, y= "transactions", data = pc)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 9)
fig.tight_layout()
plt.savefig('fig/n_transactions_store.png', format='png')

# date column to datetime
df_transactions.date = pd.to_datetime(df_transactions.date)
print(df_transactions.date.dtype)

max_transaction = df_transactions['transactions'].max()
min_transaction = df_transactions['transactions'].min()

# List of stores having the top 100 transactions

print(df_transactions.store_nbr[df_transactions['transactions'] == max_transaction])
print(df_transactions.store_nbr[df_transactions['transactions'] == min_transaction])

top_trans = df_transactions.nlargest(100, 'transactions')
print('List of stores having the top 100: ', top_trans.store_nbr.unique())

# List of stores having the top 500 transactions

top_trans = df_transactions.nlargest(500, 'transactions')
print('List of stores having the top 500: ',top_trans.store_nbr.unique())

# List of stores having the top 1000 transactions

top_trans = df_transactions.nlargest(1000, 'transactions')
print('List of stores having the top 1000: ',top_trans.store_nbr.unique())

# Memory reduction
mem = df_transactions.memory_usage(index=True).sum()
print("Memory consumed by transactions dataframe initially  :   {} MB" .format(mem/ 1024**2))

change_datatype(df_transactions)
change_datatype_float(df_transactions)

mem = df_transactions.memory_usage(index=True).sum()
print("\nMemory consumed by transactions dataframe later  :   {} MB" .format(mem/ 1024**2))


# holidays_events.csv

print('-------- holidays_events.csv ----------')
print(df_holidays_events.shape)
print(df_holidays_events.head())

print('Any missing values: ', df_holidays_events.isnull().values.any())

pp = pd.value_counts(df_holidays_events.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_holidays.png', format='png')

print(df_holidays_events.dtypes.unique())
print(df_holidays_events.dtypes.nunique())

print(df_holidays_events.type.unique())
print(df_holidays_events.type.value_counts())

fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax = sns.countplot( y="type", data=df_holidays_events, palette="RdBu")
plt.savefig('fig/holidays.png', format='png')

# Distribution of different locales of holidays

print(df_holidays_events.locale.unique())
print(df_holidays_events.locale.value_counts())

fig, ax = plt.subplots()
fig.set_size_inches(8, 7)
ax = sns.countplot( x="locale", data=df_holidays_events, palette="muted")

plt.savefig('fig/locale_holidays.png', format='png')

# Distribtuion of types of holidays with respect to various locales

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
sns.countplot( x="type", hue="locale", data=df_holidays_events, palette="muted")
plt.savefig('fig/type_locale_holidays.png', format='png')

# Dropping locale_name, description as they are not of importance for prediction hence can be dropped
df_holidays_events.drop(['locale_name', 'description'], axis=1, inplace=True)

# Distribution of holidays transferred vs non-transferred
print(df_holidays_events.transferred.value_counts())

print(df_holidays_events.transferred.hist())

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
sns.countplot( x="type", hue="transferred", data=df_holidays_events, palette="Blues_d")
plt.savefig('fig/transferred_holidays.png', format='png')

obj_cols = list(df_holidays_events.select_dtypes(include=['object']).columns)
for col in obj_cols:
    df_holidays_events[col], _ = pd.factorize(df_holidays_events[col])
    
df_holidays_events.head(10)

# Memory Reduction

mem = df_holidays_events.memory_usage(index=True).sum()
print("Memory consumed by transactions dataframe initially  :   {} MB" .format(mem/ 1024**2))

change_datatype(df_holidays_events)
change_datatype_float(df_holidays_events)

mem = df_holidays_events.memory_usage(index=True).sum()
print("\nMemory consumed by transactions dataframe later  :   {} MB" .format(mem/ 1024**2))


# oil.csv
print('-------- oil.csv --------')
print(df_oil.shape)
print(df_oil.head())

print('Any missing values: ', df_oil.isnull().values.any())

pp = pd.value_counts(df_oil.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_oil.png', format='png')

print(df_oil.dtypes.unique())
print(df_oil.dtypes.nunique())

# to datetime
df_oil.date = pd.to_datetime(df_oil.date)
print(df_oil.date.dtype)

plt.figure(figsize=(3,4))
df_oil.set_index('date').plot()

plt.savefig('fig/date_oil.png', format='png')

# train.csv
print('---------- train.csv ----------')

print(df_train.shape)
print(df_train.head())

# Memory size
mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by train dataframe initially  :   {} MB" .format(mem/ 1024**2))

# Reduce memory
change_datatype(df_train)
change_datatype_float(df_train)

# Checking Missing Values

print('Any missing values: ', df_train.isnull().values.any())

print('Missing values in : ',df_train.columns[df_train.isnull().any()].tolist())

#print('Total missing values : ', df_train.isnull().sum())

# Analyzing Datatypes
pp = pd.value_counts(df_train.dtypes)
pp.plot.bar()
#plt.show()
plt.savefig('fig/data_type_train.png',format='png')

print(df_train.dtypes.unique())
print(df_train.dtypes.nunique())

# Analysis of target variable unitsales

'''stores_unitsales = (df_train.groupby(['store_nbr']).sum())
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = stores_unitsales.index, y= "unit_sales", data = stores_unitsales)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
'''


'''
items_unitsales = (df_train.groupby(['item_nbr']).sum())
 
fig, ax = plt.subplots()
fig.set_size_inches(12, 8)
ax = sns.barplot(x = items_unitsales.index, y= "unit_sales", data = items_unitsales)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
'''


print('non NaN rows:', df_train.onpromotion.count())

