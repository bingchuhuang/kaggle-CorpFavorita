import click
import os
import sys 
sys.path.append('../')
from config import config

def run_data_preprocess():
    cmd = 'python3 ./data_preprocess.py'
    os.system(cmd)

def run_feature_extraction():
    cmd = 'python3 ./feature_extraction.py'
    os.system(cmd)

def run_combine_feature(feature_list_file):
    cmd = 'python3 ./combine_feature.py ' + feature_list_file + ' ' + config.feature_path + ' train'
    os.system(cmd)

    cmd = 'python3 ./combine_feature.py ' + feature_list_file + ' ' + config.feature_path + ' test'
    os.system(cmd)


@click.command()
@click.option('--runtype', default=None, type=str, help='PO/FO/CO/FC, preprocess/feature only/combine only/feature and combine.')
@click.option('--fset', default=None, type=str, help='feature set filename')
@click.option('--email/--no-email', default=False)
def main(runtype, fset, email):

    if runtype is None:
        print('Error: No input run type!')
        os._exit(0)
    else:
        assert runtype in {'PO','FO','CO','FC'} 

    print('runtype = ', runtype)

    if runtype == 'PO':
        run_data_preprocess()

    if runtype == 'FO':
        run_feature_extraction()

    if runtype == 'CO':
        assert fset != None
        run_combine_feature(fset)

    if runtype == 'FC':
        run_feature_extraction()
        assert fset != None
        run_combine_feature(fset)

    print('run_features.py done!')

if __name__ == '__main__':
    main()
