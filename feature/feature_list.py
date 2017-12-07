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
    'id':simple_read,

    # date 
    'date':simple_read,
    'dow':day_of_week,
    'dom':day_of_month,
    'woy':week_of_year,
    'mon':month,
    'year':year,
    
    # store
    'store_nbr':simple_read,
    'cluster':store_feature,
    'city':store_feature,
    'state':store_feature,
    'stype':store_feature, #store type

    # item
    'item_nbr':simple_read,
    'family':item_feature,
    'class':item_feature,
    'perishable':item_feature,
    'onpromotion':simple_read,

    # store x item
    
    # holiday and events
    'type':holiday_feature, #holiday type
    'locale':holiday_feature,
    'locale_name':holiday_feature,
    'transferred':holiday_feature,
    'description':holiday_feature,
    'holidays_thisweek':holiday_feature,
    'holidays_lastweek':holiday_feature,
    'holidays_nextweek':holiday_feature,
    'day_before_holiday_national':holiday_feature,
    'day_after_holiday_national':holiday_feature,
    'day_before_holiday_regional':holiday_feature,
    'day_after_holiday_regional':holiday_feature,
    'day_before_holiday_local':holiday_feature,
    'day_after_holiday_local':holiday_feature,

    # weather
    'humidity':weather_feature,
    'temperatureMin':weather_feature,
    'temperatureMax':weather_feature,

    # stats of transactions
    'tx_med':transaction_feature,
    'tx_mean':transaction_feature,
    'tx_std':transaction_feature,
    'tx_skew':transaction_feature,
    'tx_kurtosis':transaction_feature,
    'tx_hm':transaction_feature,
    'tx_10pct':transaction_feature,
    'tx_90pct':transaction_feature,

    # stats of sales
    'item_sales_med':sales_feature,
    'item_sales_mean':sales_feature,
    'item_sales_std':sales_feature,
    'item_sales_skew':sales_feature,
    'item_sales_kurtosis':sales_feature,
    #'item_sales_hm':sales_feature,
    'item_sales_10pct':sales_feature,
    'item_sales_90pct':sales_feature,

    # oil price
    'dcoilwtico':oil_feature
}
