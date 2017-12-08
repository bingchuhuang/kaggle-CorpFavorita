import os
import numpy as np

class Configure:
    def __init__(self, data_path, proc_data_path, feature_path, comb_feature_path, model_out_path):

        # raw data
        self.data_path              = data_path
        self.fname_train            = 'train_small_set1'
        #self.fname_train            = 'train'
        self.fname_test             = 'test'
        self.fname_stores           = 'stores'
        self.fname_transactions     = 'transactions'
        self.fname_items            = 'items'
        self.fname_oil              = 'oil'
        self.fname_holidays         = 'holidays_events'
        self.fname_holiday_counts   = 'holiday_counts'
        self.fname_weather          = 'weather'

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

        self.dict_dtypes = {
            'id':np.int32,
            'family':np.int16,
            'class':np.int16,
            'city':np.int16,
            'state':np.int8,
            'stype':np.int8,
            'type':np.int8,
            'locale':np.int8,
            'locale_name':np.int8,
            'description':np.int8,
            'humidity':np.float16,
            'temperatureMin':np.float16,
            'temperatureMax':np.float16,
            'year':np.int16,
            'doy':np.int16,
            'woy':np.int8,
            'transferred':np.int8,
            'onpromotion':np.int8,
            'dow':np.int8,
            'dom':np.int8,
            'holidays_thisweek':np.int8,
            'holidays_lastweek':np.int8,
            'holidays_nextweek':np.int8,
            'day_before_holiday_national':np.int16,
            'day_after_holiday_national':np.int16,
            'day_before_holiday_regional':np.int16,
            'day_after_holiday_regional':np.int16,
            'day_before_holiday_local':np.int16,
            'day_after_holiday_local':np.int16
        }

    def set_dtype(self, df):
        for name in df.columns:
            if name in self.dict_dtypes.keys():
                df[name] = df[name].astype(self.dict_dtypes[name])


config = Configure(data_path='../data',proc_data_path='../proc_data',feature_path='../single_feature',comb_feature_path='../comb_feature',model_out_path='../model/out')
