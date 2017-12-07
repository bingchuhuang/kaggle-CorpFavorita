import xgboost as xgb
import numpy as np
import pandas as pd
import operator
from matplotlib import pylab as plt


'''
Evaluation function
'''

def NWRMSLE(y, pred, weights):
    y = np.array(y).clip(0,np.max(y))
    pred = np.array(pred).clip(0,np.max(pred))
    weighted_errors = weights * np.square(np.log1p(pred) - np.log1p(y))
    return np.sqrt(np.sum(weighted_errors)/np.sum(weights))

'''
XGBoost
'''

def run_xgb(par_list, df_train, y_train, output_fname):
    param = par_list
    num_round = 20
    dtrain = xgb.DMatrix(df_train, label=y_train)
    bst = xgb.train(
        param,
        dtrain,
        num_round
    )
    bst.save_model(output_fname)

    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.gcf().savefig('feature_importance_xgb.png')

def evaluate_xgb(saved_model_fname, df_cv, y_cv):
    dcv = xgb.DMatrix(df_cv, label=y_cv)

    bst = xgb.Booster({'nthread':4})
    bst.load_model(saved_model_fname)
    preds = bst.predict(dcv)

    yt = np.array(y_cv, dtype=pd.Series)
    yt = yt.astype(float)
    logdiff = np.log(preds+1.) - np.log(yt+1.)
    logdiff = logdiff[np.isfinite(logdiff)]
    nwrmsle_nowt = np.sqrt(np.sum(logdiff**2)/logdiff.shape[0])
    print('nwrmsle_nowt = ', nwrmsle_nowt)

    weights = df_cv['perishable']*0.25 + 1

    nwrmsle_func = NWRMSLE(yt,preds,weights)
    print('nwrmsle_func = ', nwrmsle_func)

def predict_xgb(saved_model_fname, df_test):
    dtest = xgb.DMatrix(df_test)

    bst = xgb.Booster({'nthread':4})
    bst.load_model(saved_model_fname)
    preds = bst.predict(dtest)

    return preds

'''
CNN
'''

def run_cnn(par_list, df_train, df_test, output_fname):
    pass


def evaluate_cnn(saved_model_fname, df_test, y_test):
    pass

def predict_cnn(saved_model_fname, df_test):
    pass

'''
RNN
'''

def run_rnn(par_list, df_train, df_test, output_fname):
    pass

def evaluate_rnn(saved_model_fname, df_test, y_test):
    pass

def predict_rnn(saved_model_fname, df_test):
    pass
