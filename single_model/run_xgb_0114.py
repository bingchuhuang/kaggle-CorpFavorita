"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle
import gc

print('Loading Data')
#df_train = pd.read_csv(
#    '../data/train.csv', usecols=[1, 2, 3, 4, 5],
#    dtype={'onpromotion': bool},
#    converters={'unit_sales': lambda u: np.log1p(
#        float(u)) if float(u) > 0 else 0},
#    parse_dates=["date"],
#    skiprows=range(1, 64358909)  # 2016-01-01
#)
with open('../data/train_2016.pkl','rb') as f:
    df_train = pickle.load(f)


item_nbr_u = df_train[df_train.date>pd.datetime(2017,8,10)].item_nbr.unique()

#df_test = pd.read_csv(
#    "../data/test.csv", usecols=[0, 1, 2, 3, 4],
#    dtype={'onpromotion': bool},
#    parse_dates=["date"]  # , date_parser=parser
#)

with open('../data/test.pkl','rb') as f:
    df_test = pickle.load(f)

items = pd.read_csv(
    "../data/items.csv",
).set_index("item_nbr")

transactions = pd.read_csv(
    "../data/transactions.csv",
    dtype={'transactions': np.float32},
    parse_dates=['date']
)

oil = pd.read_csv(
    "../data/oil.csv",
    dtype={'dcoilwtico': np.float32},
    parse_dates=['date']
)


# holidays
df_hol = pd.read_csv(
    "../data/holiday_by_store_date.csv"
    ,parse_dates=['date'])


df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,12,26)]
del df_train
gc.collect()

hol_2017 = pd.merge(df_2017, df_hol, how='left', on=['date','store_nbr'])
df_test = pd.merge(df_test, df_hol, how='left', on=['date','store_nbr'])

df_test = df_test.set_index(
    ['store_nbr', 'item_nbr', 'date']
)

holiday_2017_train = hol_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["holiday"]].unstack(
        level=-1).fillna(False)
holiday_2017_train.columns = holiday_2017_train.columns.get_level_values(1)
holiday_2017_test = df_test[["holiday"]].unstack(level=-1).fillna(False)
holiday_2017_test.columns = holiday_2017_test.columns.get_level_values(1)
holiday_2017_test = holiday_2017_test.reindex(holiday_2017_train.index).fillna(False)
holiday_2017 = pd.concat([holiday_2017_train, holiday_2017_test], axis=1)
del holiday_2017_test, holiday_2017_train
gc.collect()

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train
gc.collect()

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)

df_2017_tmp = df_2017.stack(level=1)

df_2017_tmp.reset_index(inplace=True)

tran = pd.merge(df_2017_tmp, transactions, how='left', on=['store_nbr','date'])
tran_2017  = tran.set_index(
        ["store_nbr", "item_nbr", "date"])[['transactions']].unstack(level=-1)
tran_2017.columns = tran_2017.columns.get_level_values(1)

oil = pd.merge(df_2017_tmp, oil, how='left', on=['date'])
oil_2017  = oil.set_index(
        ["store_nbr", "item_nbr", "date"])[['dcoilwtico']].unstack(level=-1)
oil_2017.columns = oil_2017.columns.get_level_values(1)
del tran, df_2017_tmp, oil


df_2017.columns = df_2017.columns.get_level_values(1)
items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_nearwd(date,b_date):
    date_list = pd.date_range(date-timedelta(140),periods=21,freq='7D').date
    result = date_list[date_list<=b_date][-1]
    return result

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "oil_14_2017": get_timespan(oil_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
        "unpromo_16aftsum_2017":(1-get_timespan(promo_2017, t2017+timedelta(16), 16, 16)).iloc[:,1:].sum(axis=1).values, 
        "prevweek_mean_7_2017": get_timespan(df_2017, t2017-timedelta(days=7), 7, 7).mean(axis=1).values,
        "prevtwoweek_mean_7_2017": get_timespan(df_2017, t2017-timedelta(days=14), 7, 7).mean(axis=1).values,
        "prevmonth_mean_7_2017": get_timespan(df_2017, t2017-timedelta(days=30), 7, 7).mean(axis=1).values,
        "prevquarter_mean_7_2017": get_timespan(df_2017, t2017-timedelta(days=90), 7, 7).mean(axis=1).values,
        "prevhalfyear_mean_7_2017": get_timespan(df_2017, t2017-timedelta(days=180), 7, 7).mean(axis=1).values,
        "prevweek_median_7_2017": get_timespan(df_2017, t2017-timedelta(days=7), 7, 7).median(axis=1).values,
        "prevtwoweek_median_7_2017": get_timespan(df_2017, t2017-timedelta(days=14), 7, 7).median(axis=1).values,
        "prevmonth_median_7_2017": get_timespan(df_2017, t2017-timedelta(days=30), 7, 7).median(axis=1).values,
        "prevquarter_median_7_2017": get_timespan(df_2017, t2017-timedelta(days=90), 7, 7).median(axis=1).values,
        "prevhalfyear_median_7_2017": get_timespan(df_2017, t2017-timedelta(days=180), 7, 7).median(axis=1).values,
        "holiday_thisweek_2017": get_timespan(holiday_2017, t2017, 7, 7).sum(axis=1).values,
        "holiday_nextweek_2017": get_timespan(holiday_2017, t2017+timedelta(days=7), 7, 7).sum(axis=1).values,
        "holiday_next2week_2017": get_timespan(holiday_2017, t2017+timedelta(days=14), 7, 7).sum(axis=1).values,
        "holiday_lastweek_2017": get_timespan(holiday_2017, t2017-timedelta(days=7), 7, 7).sum(axis=1).values,
        "holiday_last2week_2017": get_timespan(holiday_2017, t2017-timedelta(days=14), 7, 7).sum(axis=1).values
    })

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
        for j in [14,60,140]:
            X["aft_promo_{}{}".format(i,j)] = (promo_2017[
                t2017 + timedelta(days=i)]-1).values.astype(np.uint8)
            X["aft_promo_{}{}".format(i,j)] = X["aft_promo_{}{}".format(i,j)]\
                                        *X['promo_{}_2017'.format(j)]
        if i ==15:
            X["bf_unpromo_{}".format(i)]=0
        else:
            X["bf_unpromo_{}".format(i)] = (1-get_timespan(
                    promo_2017, t2017+timedelta(16), 16-i, 16-i)).iloc[:,1:].sum(
                            axis=1).values / (15-i) * X['promo_{}'.format(i)]

    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['median_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').median(axis=1).values
        X['mean_10_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['median_10_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').median(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values        
        X['median_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').median(axis=1).values        
        X['std_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').std(axis=1).values        
        X['skew_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').skew(axis=1).values        
        X['kurt_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').kurtosis(axis=1).values        
        X['10pct_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').quantile(0.1, axis=1).values        
        X['90pct_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').quantile(0.9, axis=1).values        
        
        date = get_nearwd(t2017+timedelta(i),t2017) # last same week day sales
        ahead = (t2017-date).days
        if ahead!=0:
            X['ahead0_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead, ahead).mean(axis=1).values
            X['ahead7_{}'.format(i)] = get_timespan(df_2017, date+timedelta(ahead), ahead+7, ahead+7).mean(axis=1).values
        X["day_1_2017_{}1".format(i)]= get_timespan(df_2017, date, 1, 1).values.ravel()
        X["day_1_2017_{}2".format(i)]= get_timespan(df_2017, date-timedelta(7), 1, 1).values.ravel()
        for m in [3,7,14,30,60,140]:
            X["mean_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).mean(axis=1).values
            X["median_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).median(axis=1).values
            X["mean_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).mean(axis=1).values
            X["median_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).median(axis=1).values
            X["std_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).std(axis=1).values
            X["std_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).std(axis=1).values
            X["skew_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).skew(axis=1).values
            X["skew_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).skew(axis=1).values
            X["90pct_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).quantile(0.9,axis=1).values
            X["90pct_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).quantile(0.9,axis=1).values
            #X["kurtosis_{}_2017_{}1".format(m,i)]= get_timespan(df_2017, date,m, m).kurtosis(axis=1).values
            #X["kurtosis_{}_2017_{}2".format(m,i)]= get_timespan(df_2017, date-timedelta(7),m, m).kurtosis(axis=1).values
    X['mean_20_dowsum_2017'] = X[['mean_20_dow{}_2017'.format(i) for i in range(7)]].sum(axis=1)
    for i in range(7):
        X['mean_20_dow{}_ratio_2017'.format(i)] = X['mean_20_dow{}_2017'.format(i)]/X['mean_20_dowsum_2017']

    #for i in range(16):
    #    X["holiday_{}".format(i)] = holiday_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
    #    for j in [14,60,140]:
    #        X["aft_holiday_{}{}".format(i,j)] = (holiday_2017[t2017 + timedelta(days=i)]-1).values.astype(np.uint8)
    #        X["aft_holiday_{}{}".format(i,j)] = X["aft_holiday_{}{}".format(i,j)]*X['holiday_{}_2017'.format(j)]
    #    if i ==15:
    #        X["bf_unholiday_{}".format(i)] = 0
    #    else:
    #        X["bf_unholiday_{}".format(i)] = (1-get_timespan(
    #                holiday_2017, t2017+timedelta(16), 16-i, 16-i)).iloc[:,1:].sum(
    #                        axis=1).values / (15-i) * X['holiday_{}'.format(i)]


    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")

t2017 = date(2017, 7, 5)
X_l, y_l = [], []
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
gc.collect()
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

print("Training and predicting models...")
param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.5
param['max_depth'] = 3
param['silent'] = 1
param['eval_metric'] = 'rmse'
param['min_child_weight'] = 4
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7
param['seed'] = 1688
num_rounds = 1000
plst = list(param.items())

df_true = pd.DataFrame(y_val,index=df_2017.index,columns=pd.date_range("2017-08-16", periods=16))

val_pred = []
test_pred = []
cate_vars = []
dtest = xgb.DMatrix(X_test)
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = xgb.DMatrix(
        X_train, label=y_train[:, i],
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
    )
    dval = xgb.DMatrix(
        X_val, label=y_val[:, i],
        weight=items["perishable"] * 0.25 + 1)
        
    watchlist = [ (dtrain,'train'), (dval, 'val') ]
    model = xgb.train(plst, dtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)
    
    val_pred.append(model.predict(dval))
    test_pred.append(model.predict(dtest))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose())**0.5)

p_val= np.array(val_pred).transpose()
df_val= pd.DataFrame(
    p_val, index=df_2017.index,
    columns=pd.date_range("2017-07-26", periods=16)
).stack().to_frame("unit_sales")
df_val.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

df_val.to_csv('out/xgb_0115_pred.csv', float_format='%.4f', index=None)
df_true.to_csv('out/xgb_0115_true.csv', float_format='%.4f', index=None)

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0).reset_index()
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 10000)
submission.loc[~submission.item_nbr.isin(item_nbr_u),'unit_sales']=0
del item_nbr_u
submission[['id','unit_sales']].to_csv('xgb_0115.csv', float_format='%.4f', index=None)
