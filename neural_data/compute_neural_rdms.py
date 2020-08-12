import os, pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Sparse Random Projection + Ridge Regression')
    parser.add_argument('--reliability_threshold', type=int, required=False, default = 0.75,
                        help='number of components to extract with sparse random projection')
    
    args = parser.parse_args()
    
    reliability_threshold = args.reliability_threshold

    cell_responses = pickle.load(open('cell_response_average.pkl', 'rb'))
    cell_data = pd.read_csv(os.path.join(data_saver_dir, 'cell_data_plus.csv'))
    cell_subset = cell_data[(cell_data['p_run_mod_ns'] == False) & 
                             (cell_data['splithalf_r'] > reliability_threshold)]['cell_specimen_id'].to_list()

    brain_rdms = {}
    for area in sorted(cell_data['area'].unique()):
        brain_rdms[area] = {}
        for layer in sorted(cell_data[cell_data['area'] == area]['layer'].unique()):
            neural_site_subset = cell_data[(cell_data['area'] == area) & (cell_data['layer'] == layer)]['cell_specimen_id']
            target_cells = [cell for cell in neural_site_subset if cell in cell_subset]
            target_cell_responses = np.stack([cell_responses[cell] for cell in target_cells]).transpose()
            brain_rdms[area][layer] = np.corrcoef(target_cell_responses)