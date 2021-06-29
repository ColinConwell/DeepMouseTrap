import os, sys
import numpy as np
import pandas as pd
from PIL import Image

def iterative_subset(df, index, cat_list):
    out_dict = {}
    df = df[[index] + cat_list]
    for row in df.itertuples():
        current = out_dict
        for i, col in enumerate(cat_list):
            cat = getattr(row, col)
            if cat not in current:
                if i+1 < len(cat_list):
                    current[cat] = {}
                else:
                    current[cat] = []
            current = current[cat]
            if i+1 == len(cat_list):
                current += [getattr(row, index)]
    
    return out_dict

def get_rdm_by_subset(groups, responses):
    rdm_dict = {}
    def get_neural_rdm(rdm_dict, groups):
        for key, value in groups.items():
            if isinstance(value, dict):
                rdm_dict[key] = {}
                get_neural_rdm(rdm_dict[key], groups[key])
            if isinstance(value, list):
                rdm_dict[key] = np.corrcoef(responses.loc[value,:].to_numpy().transpose())
                
    get_neural_rdm(rdm_dict, groups)
            
    return rdm_dict

class AllenBrainObservatory:
    def __init__(self, cell_subset_args = None, dataset_path = 'brain_observatory'):
        
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(path_dir, 'brain_observatory')
        
        images_path = os.path.join(data_dir, 'stimulus_set.npy')
        metadata_path = os.path.join(data_dir, 'cell_metadata_mini.csv')
        response_path = os.path.join(data_dir, 'cell_response_average.csv')
            
        self.stimulus_set = np.load(images_path)
        self.n_stimuli = len(self.stimulus_set)
        
        self.cell_responses = pd.read_csv(response_path).set_index('cell_specimen_id')
        self.cell_metadata = pd.read_csv(metadata_path).set_index('cell_specimen_id')
        
        if cell_subset_args is not None:
            self.get_cell_subset(cell_subset_args, return_subset = False, apply_subset = True)
        
    def get_cell_subset(self, cell_subset_args, return_subset = False, apply_subset = False):
        cells = self.cell_metadata

        if 'image_selective' in cell_subset_args:
            cells = cells[cells['image_selective'] == cell_subset_args['image_selective']]

        if 'splithalf_r' in cell_subset_args:
            cells = cells[cells['splithalf_reliability'] >= cell_subset_args['splithalf_r']]
            
        if 'significant_running' in cell_subset_args:
            cells = cells[(cells['significant_running'] == cell_subset_args['significant_running'])]

        self.cell_subset = cells.index.to_list()

        if apply_subset:
            self.cell_responses = self.cell_responses.loc[self.cell_subset,:]
            self.cell_metadata = self.cell_metadata.loc[self.cell_subset,:]
        
        if return_subset:
            return self.cell_subset
            
    def view_sample_stimulus(self, image_index = None):
        if image_index is None:
            image_index = np.random.randint(self.n_stimuli)
        return Image.fromarray(self.stimulus_set[image_index])
    
    def get_neural_rdms(self, group_vars, subset = None, index = 'cell_specimen_id', responses = None):
        cell_data = self.cell_metadata.loc[subset,:] if subset is not None else self.cell_metadata
        cell_groups = iterative_subset(cell_data.reset_index(), index, group_vars)
        cell_responses = self.cell_responses if responses is None else responses
        return get_rdm_by_subset(cell_groups, cell_responses)
        
    @staticmethod
    def get_cell_subset_options():
        return {'signficant_running': 'return cells significantly modulated by running',
                'image_selective': 'return cells significantly modulated by natural scenes',
                'splithalf_r': 'return cells past a set threshold of splithalf reliability'}