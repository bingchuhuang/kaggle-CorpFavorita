import sys
import click
import model_utils as mod
sys.path.append('../')
from config import config
from os import path, _exit
import pickle
from datetime import date
import pandas as pd
import numpy as np
import configparser 
import time

def read_config(f):
    conf = configparser.ConfigParser()
    conf.read(f)
    model = conf['model']['method']
    par_list = conf._sections['param']
    feature_conf = conf._sections['feature']['feature_set']
    return model, par_list, feature_conf

def combine_feature(fset_file, feature_path, tag):
    print('combining features ...')

    feature_names = [ x.rstrip(' \n') for x in open(fset_file)]
    print('feature names = ', feature_names)
    x_list = []
    for feature in feature_names:
        fpkl = path.join(feature_path,tag+'_'+feature+'.pkl')
        with open(fpkl,'rb') as f:
            x = pickle.load(f)
        x_list.append(x)
    
    if tag == 'train':
        fpkl = path.join(feature_path,tag+'_unit_sales.pkl')
        with open(fpkl,'rb') as f:
            x = pickle.load(f)
        x_list.append(x)
        feature_names.append('unit_sales')
    
    x_comb = pd.concat(x_list,axis=1,keys=feature_names)
   
    print('Features combined!')
    return x_comb

@click.command()
@click.option('--mode', default=None, type=str, help='T/E/P, train / evaluate / predict')
@click.option('--conf', default=None, type=str, help='conf file name')
@click.option('--email/--no-email', default=False)
def main(mode, conf, email):
   
    if mode is None:
        print('Error: No input run type!')
        _exit(0)
    else:
        assert mode in {'T','E','P','TE','EP','TEP'}

    if conf is None:
        print('Error: No input conf!')
        _exit(0)
    
    print('config_file : ',conf)

    model, par_list, feature_conf = read_config(conf)
    print('model        = ', model)
    print('par_list     = ', par_list)
    print('feature_conf = ', feature_conf)

    fset_file = path.join('./fset',feature_conf)
    df_train = combine_feature(fset_file, config.feature_path, 'train')
    
    #nr = df_train.shape[0]
    #nsplit = int(nr*0.7)
    #df_cv = df_train[:nsplit]
    #df_train = df_train[nsplit:]

    df_train.drop(df_train[df_train['unit_sales']<0].index,inplace=True)
    row  = df_train[df_train['date']=='2017-08-01'].index[0]
    df_cv = df_train[row:]
    df_train = df_train[:row-1]

    y_cv = df_cv['unit_sales']
    y_train = df_train['unit_sales']

    df_ev = df_cv.copy()

    df_cv = df_cv.drop('unit_sales', axis=1)
    df_train = df_train.drop('unit_sales', axis=1)
    df_cv = df_cv.drop('date', axis=1)
    df_train = df_train.drop('date', axis=1)

    print('df_train')
    print(df_train.head(5))
    
    print('y_train')
    print(y_train.head(5))
    print(df_train.dtypes)
    print(y_train.dtype)

    tmp = time.time()
    if 'T' in mode:
        outname = path.join(config.model_out_path,model+'_'+feature_conf+'.bin')
        if model == 'xgb':
            mod.run_xgb(par_list,df_train,y_train,outname)
        
        if model == 'cnn':
            mod.run_cnn(par_list,df_train,y_train,outname)

        if model == 'rnn':
            mod.run_rnn(par_list,df_train,y_train,outname)

    train_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(train_time)))

    tmp = time.time()
    
    if 'E' in mode:
        model_path = path.join(config.model_out_path,model+'_'+feature_conf+'.bin')
        if model == 'xgb':
            mod.evaluate_xgb(model_path, df_cv, y_cv)

        if model == 'cnn':
            mod.evaluate_cnn(model_path, df_cv, y_cv)
        
        if model == 'rnn':
            mod.evaluate_rnn(model_path, df_cv, y_cv)
        df_ev['pred'] = y_cv
        with open('df_evaluation_'+feature_conf+'.pkl','wb') as f:
            pickle.dump(df_ev,f,-1)
         
    eval_time = time.time() - tmp
    print("CPU evaluation Time: %s seconds" % (str(eval_time)))

    if 'P' in mode:
        df_test = combine_feature(fset_file, config.feature_path, 'test')
        with open(path.join(config.feature_path,'test_id.pkl'),'rb') as f:
            df_test_id = pickle.load(f)
        df_test = df_test.drop('date', axis=1)
        model_path = path.join(config.model_out_path,model+'_'+feature_conf+'.bin')
        df_test['id'] = df_test_id
        
        if model == 'xgb':
            y_pred = mod.predict_xgb(model_path, df_test)
            print(y_pred)

        if model == 'cnn':
            y_pred = mod.predict_cnn(model_path, df_test)
            print(y_pred)
        
        if model == 'rnn':
            y_pred = mod.predict_rnn(model_path, df_test)
            print(y_pred)
        df_test['unit_sales'] = np.around(y_pred,decimals=1)
        out_name = 'submission_' + feature_conf + '.csv'
        df_test[['id','unit_sales']].to_csv(out_name,index=False)


if __name__ == '__main__':
    main()
