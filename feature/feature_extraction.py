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
#import pandas as pd
sys.path.append('../')
from config import config
from feature_list import feature_list
from feature_utils import simple_read 
from os import path, makedirs

def dump_feature(df, func, feature,ftype):
    X = func(df,feature)
    with open(path.join(config.feature_path,ftype+'_'+feature+'.pkl'),'wb') as f:
        pickle.dump(X, f, -1)

if __name__ == '__main__':
    
    # load data

    with open(path.join(config.proc_data_path,config.fname_train+'.pkl'),'rb') as f:
        df_train = pickle.load(f)
    with open(path.join(config.proc_data_path,config.fname_test+'.pkl'),'rb') as f:
        df_test = pickle.load(f)

    # generate features
    print('feature_extraction: generating features ... ')
    
    if not path.exists(config.feature_path):
        makedirs(config.feature_path)
    for key, value in feature_list.items():
        print('feature = ', key) 
        dump_feature(df_train,value,str(key),'train')
        dump_feature(df_test,value,str(key),'test')
    
    dump_feature(df_train,simple_read,'unit_sales','train')

    print('feature_extraction: Mission accomplished!') 
