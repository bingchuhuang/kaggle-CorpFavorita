import numpy as np
from scipy import stats

def compute_stats(df, print_flag=False):
    '''
    compute statistics for an array
    '''
    mean        = np.mean(df)
    median      = np.median(df) 
    std         = np.std(df)
    skew        = stats.skew(df,bias=False)
    kurtosis    = stats.kurtosis(df)
    percent10   = np.percentile(df,10)
    percent90   = np.percentile(df,90)

    if print_flag :
        print('median       = ', median   )
        print('mean         = ', mean     )
        print('std          = ', std      )
        print('skew         = ', skew     )
        print('kurtosis     = ', kurtosis )
        print('percentile10 = ', percent10)
        print('percentile90 = ', percent90)
    return median, mean, std, skew, kurtosis, percent10, percent90
