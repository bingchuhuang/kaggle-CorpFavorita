from os import path, makedirs
import pickle
import pandas as pd
import sys 
sys.path.append('../')
from config import config

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
    
    if not path.exists(config.comb_feature_path):
        makedirs(config.comb_feature_path)

    print(x_comb.head(5))
    fname = path.basename(fset_file)
    with open(path.join(config.comb_feature_path,tag+'_'+fname+'.pkl'),'wb') as f:
        pickle.dump(x_comb, f, -1)

if __name__ == '__main__':
    combine_feature(sys.argv[1],sys.argv[2],sys.argv[3])
