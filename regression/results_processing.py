import os, sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Regression Results Processing')
    parser.add_argument('--input_dir', required=True, type=str,
                        help='directory with regression results')
    parser.add_argument('--output_dir', type=str, required=False, default='./',
                        help='directory for output from random projections')
    parser.add_argument('--target_output', type=str, required=False, default='./',
                        help='directory for output from random projections')


source_dir = 'nu_srp_ridge_results'

asset_dflist = []
for asset_dir in tqdm(os.listdir(output_dir)):
    if 'taskonomy' not in asset_dir:
        incoming_assets = glob(os.path.join(output_dir, asset_dir + '/*.csv'))
        for asset in tqdm(incoming_assets, leave=False):
            asset_dflist.append(pd.read_csv(asset))
    
scoresheet = pd.concat(asset_dflist)