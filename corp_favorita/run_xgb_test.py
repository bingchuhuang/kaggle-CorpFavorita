import xgboost as xgb
import pandas as pd
import numpy as np


def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['onpromotion'] = df['onpromotion'].fillna(-999)
    #df = df.fillna(-999)
    #df['onpromotion'] = df['onpromotion'].astype(int)
    df['onpromotion'] = df['onpromotion'].astype(int)
    df = df.drop('date', axis=1)

#df_train = pd.read_csv('../data/train.csv')
df_train = pd.read_csv('../data/train_small_set.csv')
df_test = pd.read_csv('../data/train_small_set.csv')
#df_test = pd.read_csv('../data/test_small_set.csv')
df_sample = pd.read_csv('../data/sample_submission.csv')
df_stores = pd.read_csv('../data/stores.csv')
df_items = pd.read_csv('../data/items.csv')
df_transactions = pd.read_csv('../data/transactions.csv')
df_oil = pd.read_csv('../data/oil.csv')
df_holidays_events = pd.read_csv('../data/holidays_events.csv')

print('df_train head : ')
print(df_train.head(5))

feature_columns = 'year,month,day,store_nbr,item_nbr,onpromotion'.split(',')
target_column = 'unit_sales'
preprocess(df_train)
X_train = df_train[feature_columns]
y_train = df_train[target_column]

print(' training set head : ')
print(X_train.head(5))
print(y_train.head(5))


preprocess(df_test)
X_test = df_test[feature_columns]
y_test = df_test[target_column]

print(' test set head : ')
print(X_test.head(5))
print(y_test.head(5))

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
	'objective'           : "reg:linear", 
    'booster'             : "gbtree",
    'eta'                 : 0.02, # 0.06, #0.01,
    'max_depth'           : 10, #changed from default of 8
    'subsample'           : 0.9, # 0.7
    'colsample_bytree'    : 0.7 # 0.7
    # 'num_parallel_tree'  : 2
    # 'alpha'             : 0.0001, 
    # 'lambda'            : 1
	}

num_round = 20  # the number of training iterations

bst = xgb.train( 
	param, 
	dtrain, 
	num_round
	#verbose_eval           = 0, 
	#early_stopping_rounds  = 100
	#watchlist             = watchlist,
	#maximize               = False
	)

bst.dump_model('dump.raw.txt')

preds = bst.predict(dtest)

score = np.sqrt((((y_test - preds)/y_test)**2).sum())
print('score = ', score)

with open('predictions.csv', 'w') as f:
    f.write('id,unit_sales\n')
    for i, record in enumerate(y_test):
        f.write('{},{},{}\n'.format(i+1, format(preds[i],'.2f'), record))
