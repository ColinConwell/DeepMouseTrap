import os, sys, argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm as tqdm

def max_transform(df, group_vars, measure_var = 'score', deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = list(group_vars)
    
    max_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(max) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        max_df = max_df[~max_df.duplicated(group_vars + [measure_var])]
        
    return max_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Regression Results Processing')
    parser.add_argument('--input_dir', required=True, type=str,
                        help='directory with regression results')
    parser.add_argument('--output_dir', type=str, required=False, default='./',
                        help='directory for output from random projections')
    parser.add_argument('--target_output', type=str, required=False, default='max',
                        help='directory for output from random projections')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_output = args.target_output

    print('Loading results...')
    asset_dflist = []
    for asset_dir in tqdm(os.listdir(input_dir)[:3]):
        incoming_assets = glob(os.path.join(input_dir, asset_dir + '/*.csv'))
        for asset in tqdm(incoming_assets, leave=False):
            asset_dflist.append(pd.read_csv(asset))

    print('Concatenating results...')
    scoresheet = pd.concat(asset_dflist)
    
    print('Parsing results by {}...'.format(target_output))
    
    if target_output == 'max':
        output_df = max_transform(scoresheet, group_vars = ['cell_specimen_id', 'model', 'train_type'])
        output_df.to_csv(os.path.join(output_dir, '_'.join(input_dir.split('/')) + '.csv'), index = None)
        
        