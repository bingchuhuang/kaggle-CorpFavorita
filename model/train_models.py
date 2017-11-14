import sys
import click
import model_utils as mod
sys.path.append('../')
from config import config
from os import path, _exit
import pickle
from datetime import date
import pandas as pd
import configparser 
import time

def read_config(f):
    conf = configparser.ConfigParser()
    conf.read(f)
    model = conf['model']['method']
    par_list = conf._sections['param']
    feature_conf = conf._sections['feature']['feature_set']
    return model, par_list, feature_conf

@click.command()
@click.option('--mode', default=None, type=str, help='T/E/TE, train or evaluate or both')
@click.option('--conf', default=None, type=str, help='conf file name')
@click.option('--email/--no-email', default=False)
def main(mode, conf, email):
   
    if mode is None:
        print('Error: No input run type!')
        _exit(0)
    else:
        assert mode in {'T','E','P','TE','EP','TEP'}
    
    print('config_file : ',conf)

    model, par_list, feature_conf = read_config(conf)
    print('model        = ', model)
    print('par_list     = ', par_list)
    print('feature_conf = ', feature_conf)

    file_path = path.join(config.comb_feature_path,'train_'+feature_conf+'.pkl')
    print('file_path = ', file_path)
    with open(file_path,'rb') as f:
        df_train = pickle.load(f)
    
    nr = df_train.shape[0]
    nsplit = int(nr*0.7)

    #split_date = date(2017,1,31)
    #df_cv = df_train[df_train['date'].apply(pd.to_datetime) > split_date]
    #df_train = df_train[df_train['date'].apply(pd.to_datetime) <= split_date]

    df_cv = df_train[:nsplit]
    df_train = df_train[nsplit:]

    y_cv = df_cv['unit_sales']
    y_train = df_train['unit_sales']

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
    
    eval_time = time.time() - tmp
    print("CPU evaluation Time: %s seconds" % (str(eval_time)))

    if 'P' in mode:
        with open(path.join(config.comb_feature_path,'test_'+feature_conf+'.pkl'),'rb') as f:
            df_test = pickle.load(f)
        df_test = df_test.drop('date', axis=1)
        model_path = path.join(config.model_out_path,model+'_'+feature_conf+'.bin')
        
        if model == 'xgb':
            y_pred = mod.predict_xgb(model_path, df_test)
            print(y_pred)

        if model == 'cnn':
            y_pred = mod.predict_cnn(model_path, df_test)
            print(y_pred)
        
        if model == 'rnn':
            y_pred = mod.predict_rnn(model_path, df_test)
            print(y_pred)


if __name__ == '__main__':
    main()
