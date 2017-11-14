'''
__file__
    feature_utils.py

__description__
    functions for computing features 

__author__
    Bingchu Huang

'''

import pandas as pd
from datetime import date

def day_of_week(df,feature):
    df[feature] = pd.to_datetime(df['date']).apply(date.weekday)
    return df[feature]

def simple_read(df,feature):
    return df[feature]
