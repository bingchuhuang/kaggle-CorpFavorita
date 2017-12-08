import click
import os
import sys 
sys.path.append('../')
from config import config

def run_data_preprocess():
    cmd = 'python3 ./data_preprocess.py'
    os.system(cmd)

def run_feature_extraction(name=None):
    if name:
        cmd = 'python3 ./feature_extraction.py ' + name
    else:
        cmd = 'python3 ./feature_extraction.py '

    os.system(cmd)

@click.command()
@click.option('--runtype', default=None, type=str, help='P/F, preprocess/feature')
@click.option('--fname', default=None, type=str, help='single feature name')
@click.option('--email/--no-email', default=False)
def main(runtype, fname, email):

    if runtype is None:
        print('Error: No input run type!')
        os._exit(0)
    else:
        assert runtype in {'P','F','PF'} 

    print('runtype = ', runtype)

    if runtype == 'P':
        run_data_preprocess()

    if runtype == 'F':
        run_feature_extraction(fname)
    
    if runtype == 'PF':
        run_data_preprocess()
        run_feature_extraction(fname)

    print('run_features.py done!')

if __name__ == '__main__':
    main()
