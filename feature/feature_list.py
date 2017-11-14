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
    'date':simple_read,
    'store_nbr':simple_read,
    'item_nbr':simple_read,
    'onpromotion':simple_read,
    'dow':day_of_week
}
