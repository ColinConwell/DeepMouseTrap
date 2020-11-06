import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, argparse, pickle

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

neural_data_dir = '../neural_data/'
model_opts_dir = '../model_opts'

sys.path.append(model_opts_dir)
from feature_extraction import *
from srp_extraction import *
from neural_regression import *
from model_options import *

# MouseLand Sparse Random Projection + Ridge Regression for PyTorch Zoo

def max_transform(df, group_vars, measure_var = 'score', deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = list(group_vars)
    
    max_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(max) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        max_df = max_df[~max_df.duplicated(group_vars + [measure_var])]
        
    return max_df

def parse_regression_results(results_dir, results_string, parse_type, parsed_results_dir):
    results_out = '{}/{}_{}.csv'.format(parsed_results_dir, results_dir, results_string, parse_type)
    results_in = '{}/{}'.format(results_dir, results_string)

    if parse_type == 'full':
        if os.path.exists(results_out) and recompute==False:
            parsed_results = pd.read_csv(results_out)
            
        if not os.path.exists(results_out) or recompute==True:
            asset_dflist = []
            for asset_dir in tqdm(os.listdir(results_dir)):
                incoming_assets = glob(os.path.join(results_dir, asset_dir + '/*.csv'))
                for asset in tqdm(incoming_assets, leave=False):
                    asset_dflist.append(pd.read_csv(asset))

            parsed_results = pd.concat(asset_dflist)
            parsed_results.to_csv(results_out, file = None)

    if parse_type == 'max':
        
        if os.path.exists(results_out) and recompute==False:
            parsed_results = pd.read_csv(output_file)

        if not os.path.exists(output_file) or recompute==True:
            max_asset_dflist = []
            for asset_dir in tqdm(os.listdir(input_dir)):
                incoming_assets = glob(os.path.join(input_dir, asset_dir + '/*.csv'))
                incoming_asset_dflist = []
                for asset in tqdm(incoming_assets, leave=False):
                    incoming_asset_dflist.append(pd.read_csv(asset))
                if len(incoming_asset_dflist) > 0:  
                    incoming_asset_df = pd.concat(incoming_asset_dflist)
                    incoming_asset_df_max = max_transform(incoming_asset_df, 
                        group_vars = ['cell_specimen_id', 'model', 'train_type'])
                    max_asset_dflist.append(incoming_asset_df_max)

            parsed_results = pd.concat(max_asset_dflist)
            parsed_results.to_csv(results_out, index = None)
            
    return(parsed_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sparse Random Projection + Ridge Regression')
    parser.add_argument('--results_dir', required=True, type=str,
                        help='directory of regression results')
    parser.add_argument('--results_string', required=True, type=str,
                        help='subdirectory of regression results to parse')
    parser.add_argument('--recompute', required=False, type=bool,
                        help='whether or not to parse results again')
    parser.add_argument('--parse_type', required=False, choices = ['max', 'full'],
                        default = 'max', help='type of parsing to perform on results')
    parser.add_argument('--parsed_results_dir', required=False, type=str,
                        default = 'parsed_results', help='directory for parsed results')
    
    args = parser.parse_args()
    results_dir = args.results_dir
    results_string = args.result_string
    parse_type = args.parse_type
    parsed_results_dir = args.parsed_results_dir
    
    parsed_regression_results = parse_regression_results()

        