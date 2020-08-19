import os, pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

def compute_neural_rdms(cell_data, cell_subset, cell_responses):
    neural_rdms = {}
    for area in sorted(cell_data['area'].unique()):
        neural_rdms[area] = {}
        for layer in sorted(cell_data[cell_data['area'] == area]['layer'].unique()):
            neural_site_subset = cell_data[(cell_data['area'] == area) & (cell_data['layer'] == layer)]['cell_specimen_id']
            target_cells = [cell for cell in neural_site_subset if cell in cell_subset]
            target_cell_responses = np.stack([cell_responses.loc[cell,:].to_numpy() for cell in target_cells]).transpose()
            neural_rdms[area][layer] = np.corrcoef(target_cell_responses)
            
    return neural_rdms

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compute Neural RDMs')
    parser.add_argument('--reliability_threshold', type=int, required=False, default = 0.75,
                        help='threshold of reliability for subselecting neurons')
    
    args = parser.parse_args()
    reliability_threshold = args.reliability_threshold

    cell_responses = pd.read_csv(brain_data_dir + 'cell_response_average.csv').set_index('cell_specimen_id')
    cell_data = pd.read_csv(brain_data_dir + 'cell_data_plus.csv')
    cell_subset = cell_data[((cell_data['p_run_mod_ns'].isna()) | 
                            (cell_data['p_run_mod_ns'] > 0.05)) & 
                            (cell_data['p_ns'] < 0.05) &
                            (cell_data['splithalf_r'] > reliability_threshold)]['cell_specimen_id'].to_list()
    cell_response_subset = cell_responses.loc[cell_subset,:]
    
    neural_rdms = compute_neural_rdms(cell_data, cell_subset, cell_response_subset)
    
    
    
    