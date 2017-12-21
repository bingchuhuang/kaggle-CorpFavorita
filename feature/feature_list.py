'''
__file__
    feature_list.py

__description__
    list of all features 

__author__
    Bingchu Huang

'''


import sys
sys.path.append('../')
#from config import config

from feature_utils import *

#if __name__ == '__main__':
feature_list = {
    
    # id only for submission
    #'id':simple_read,

    ## date 
    #'date':simple_read,
    #'dow':day_of_week,
    #'dom':day_of_month,
    #'woy':week_of_year,
    #'mon':month,
    #'year':year,
    #
    ## store
    #'store_nbr':simple_read,
    #'cluster':store_feature,
    #'city':store_feature,
    #'state':store_feature,
    #'stype':store_feature, #store type

    ## item
    #'item_nbr':simple_read,
    #'family':item_feature,
    #'class':item_feature,
    #'perishable':item_feature,
    #'onpromotion':simple_read,

    ## store x item
    #
    ## holiday and events
    #'type':holiday_feature, #holiday type
    #'locale':holiday_feature,
    #'locale_name':holiday_feature,
    #'transferred':holiday_feature,
    #'description':holiday_feature,
    #'holidays_thisweek':holiday_feature,
    #'holidays_lastweek':holiday_feature,
    #'holidays_nextweek':holiday_feature,
    #'day_before_holiday_national':holiday_feature,
    #'day_after_holiday_national':holiday_feature,
    #'day_before_holiday_regional':holiday_feature,
    #'day_after_holiday_regional':holiday_feature,
    #'day_before_holiday_local':holiday_feature,
    #'day_after_holiday_local':holiday_feature,

    ## weather
    #'humidity':weather_feature,
    #'temperatureMin':weather_feature,
    #'temperatureMax':weather_feature,

    # stats of transactions
    #'tx_dow_med':transaction_feature,
    #'tx_dow_mean':transaction_feature,
    #'tx_dow_std':transaction_feature,
    #'tx_dow_skew':transaction_feature,
    #'tx_dow_kurtosis':transaction_feature,
    #'tx_dow_hm':transaction_feature,
    #'tx_dow_10pct':transaction_feature,
    #'tx_dow_90pct':transaction_feature,

    # stats of sales
    #'item_sales_med':sales_feature,
    #'item_sales_mean':sales_feature,
    #'item_sales_std':sales_feature,
    #'item_sales_skew':sales_feature,
    #'item_sales_kurtosis':sales_feature,
    ##'item_sales_hm':sales_feature,
    #'item_sales_10pct':sales_feature,
    #'item_sales_90pct':sales_feature,
    #
    #'item_sales_dow_med':sales_feature,
    #'item_sales_dow_mean':sales_feature,
    #'item_sales_dow_std':sales_feature,
    #'item_sales_dow_skew':sales_feature,
    #'item_sales_dow_kurtosis':sales_feature,
    #'item_sales_dow_hm':sales_feature,
    #'item_sales_dow_10pct':sales_feature,
    #'item_sales_dow_90pct':sales_feature,

    ## MA
    #'ma_is_tot':ma_feature,
    #'ma_is_112':ma_feature,
    #'ma_is_56':ma_feature,
    #'ma_is_28':ma_feature,
    #'ma_is_14':ma_feature,
    #'ma_is_7':ma_feature,
    #'ma_is_3':ma_feature,
    #'ma_is_1':ma_feature,
    #'ma_cs_tot':ma_feature,
    #'ma_cs_112':ma_feature,
    #'ma_cs_56':ma_feature,
    #'ma_cs_28':ma_feature,
    #'ma_cs_14':ma_feature,
    #'ma_cs_7':ma_feature,
    #'ma_cs_3':ma_feature,
    #'ma_cs_1':ma_feature,
    #'ma_fs_tot':ma_feature,
    #'ma_fs_112':ma_feature,
    #'ma_fs_56':ma_feature,
    #'ma_fs_28':ma_feature,
    #'ma_fs_14':ma_feature,
    #'ma_fs_7':ma_feature,
    #'ma_fs_3':ma_feature,
    #'ma_fs_1':ma_feature,
    
    'ma_is_wt':ma_wt
  
    ## oil price
    #'dcoilwtico':oil_feature
}
