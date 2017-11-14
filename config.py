import os
#import numpy as np

class Configure:
    def __init__(self, data_path, proc_data_path, feature_path, comb_feature_path, model_out_path):
        
        # raw data
        self.data_path          = data_path
        self.fname_train        = 'train_small_set'
        self.fname_test         = 'test_small_set'
        self.fname_stores       = 'stores'
        self.fname_transactions = 'transactions'
        self.fname_items        = 'items'
        self.fname_oil          = 'oil'
        self.fname_holidays     = 'holidays_events'

        # processed data
        self.proc_data_path     = proc_data_path 

        # features
        self.feature_path       = feature_path
        
        # combined features
        self.comb_feature_path  = comb_feature_path

        # model
        self.model_out_path  = model_out_path 

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        
        if not os.path.exists(self.proc_data_path):
            os.makedirs(self.proc_data_path)
        
        if not os.path.exists(self.comb_feature_path):
            os.makedirs(self.comb_feature_path)
        
        if not os.path.exists(self.model_out_path):
            os.makedirs(self.model_out_path)

config = Configure(data_path='../data',proc_data_path='../proc_data',feature_path='../single_feature',comb_feature_path='../comb_feature',model_out_path='../model/out')
