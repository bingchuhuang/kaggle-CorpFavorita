'''
__file__
    feature_extraction.py

__description__
    extract features from preprocessed data

__author__
    Bingchu Huang

'''

import sys
import pickle
sys.path.append('../')
from config import config
from feature_list import feature_list
from feature_utils import simple_read 
from os import path, makedirs
from datetime import date

def dump_feature(df, func, feature,ftype):
    X = func(df,feature)
    config.set_dtype(X)
    with open(path.join(config.feature_path,ftype+'_'+feature+'.pkl'),'wb') as f:
        pickle.dump(X, f, -1)

def dump_feature_test(df, df_train, func, feature, ftype, isFromTrain):
    if isFromTrain:
        #df_train['date'] = pd.to_datetime(df_train['date'])
        df_train = df_train[df_train['date']>date(2016,8,1)]
        X = func(df,df_train,feature)
    else:
        X = func(df,feature)
    config.set_dtype(X)
    with open(path.join(config.feature_path,ftype+'_'+feature+'.pkl'),'wb') as f:
        pickle.dump(X, f, -1)

def main(fname=None):
    # load data
    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    with open(path.join(config.proc_data_path,config.fname_test+'.pkl'),'rb') as f:
        df_test = pickle.load(f)

    # generate features
    print('feature_extraction: generating features ... ')
    print('train:')
    print(df_train.head(5))
    
    feature_from_train_list = ['shopavg_transaction','shopavg_holiday','shopavg_promo','shopavg_dow','shopavg_dom','prev_quarter_dps_mean']

    if not path.exists(config.feature_path):
        makedirs(config.feature_path)
        
    for key, value in feature_list.items():
        isFromTrain = False
        if key in feature_from_train_list or 'prev' in key:
            isFromTrain = True
        if fname is not None:
            if key==fname:
                print('feature = ', key) 
                dump_feature(df_train,value,str(key),'train')
                dump_feature(df_test,value,str(key),'test')
                #dump_feature_test(df_test,df_train,value,str(key),'test',isFromTrain)
        else:
            print('feature = ', key) 
            dump_feature(df_train,value,str(key),'train')
            dump_feature(df_test,value,str(key),'test')
            #dump_feature_test(df_test,df_train,value,str(key),'test',isFromTrain)
    
    dump_feature(df_train,simple_read,'unit_sales','train')

    print('feature_extraction: Mission accomplished!') 

if __name__ == '__main__':
    if len(sys.argv) > 1 : 
        main(sys.argv[1])
    else:
        main()
