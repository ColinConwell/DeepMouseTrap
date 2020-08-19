import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, argparse, pickle
from scipy.stats import pearsonr

neural_data_dir = '../neural_data'
model_opts_dir = '../model_opts'

sys.path.append(neural_data_dir)
from compute_neural_rdms import *

sys.path.append(model_opts_dir)
from feature_extraction import *
from rdm_extraction import *
from model_options import *

from nnls_rs_regression import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Representational Similarity Analysis')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--rsa_subtype', type=str, required=False, default='nnls-rsa',
                        help='(root) directory for all outputs')
    parser.add_argument('--output_dir', type=str, required=False, default='./',
                        help='(root) directory for all outputs')
    parser.add_argument('--results_dir', type=str, required=False, default='rsa_results',
                        help='(sub) directory for results')
    parser.add_argument('--cuda_device', type=int, required=False, default=0,
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_string = args.model_string
    rsa_subtype = args.rsa_subtype
    output_dir = args.output_dir
    results_dir = args.results_dir
    cuda_devicce = args.cuda_device
    
    results_dir = os.path.join(output_dir, results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']
    
    method_printouts = {'rsa': 'standard representational similarity', 
                        'nnls-rsa': 'nonnegative least squares representational similiarty'}
        
    print('Computing {} to {} {}; saving results into {}...'
          .format(method_printouts[rsa_subtype], model_name, get_training_printouts(train_type), results_dir))
    
    results_file = os.path.join(results_dir, model_string + '_{}.csv'.format(rsa_subtype))
    if not os.path.exists(results_file):
        
        cell_responses = pd.read_csv(neural_data_dir + '/cell_response_average.csv').set_index('cell_specimen_id')
        cell_data = pd.read_csv(neural_data_dir + '/cell_data_plus.csv')
        cell_subset = cell_data[((cell_data['p_run_mod_ns'].isna()) | 
                                (cell_data['p_run_mod_ns'] > 0.05)) & 
                                (cell_data['p_ns'] < 0.05) &
                                (cell_data['splithalf_r'] > 0.75)]['cell_specimen_id'].to_list()
        cell_response_subset = cell_responses.loc[cell_subset,:]
        
        neural_rdms = compute_neural_rdms(cell_data, cell_subset, cell_response_subset)
     
        stimulus_set = np.load('../stimulus_set.npy')
        image_transforms = get_image_transforms()['imagenet_from_numpy']
        
        model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set]))
        model_rdms = rdm_extraction(model_string, model_inputs = model_inputs, output_dir = os.path.join(output_dir, 'rdm_arrays'))
        
        if rsa_subtype == 'rsa':
            rsa_dict_list = []
            for model_layer in tqdm(model_rdms):
                model_layer_rdm = model_rdms[model_layer]
                for area in tqdm(neural_rdms.keys(), leave = False):
                    for layer in tqdm(neural_rdms[area].keys(), leave = False):
                        neural_rdm = neural_rdms[area][layer]
                        neural_rdm_triu = neural_rdm[np.triu_indices(neural_rdm.shape[0], k=1)]
                        model_layer_rdm_triu = model_layer_rdm[np.triu_indices(model_layer_rdm.shape[0], k=1)]
                        score = pearsonr((neural_rdm_triu).flatten(), (model_layer_rdm_triu).flatten())[0]**2
                        rsa_dict_list.append({'model': model_name, 'train_type': train_type, 'model_layer': model_layer, 
                                              'area': area, 'layer': layer, 'score': score, 'method': rsa_subtype})

            pd.DataFrame(rsa_dict_list).to_csv(results_file, index = None)
            
        if rsa_subtype == 'nnls-rsa':
            model_rdm_stack = np.stack([model_rdms[model_layer] 
                                        for model_layer in list(model_rdms.keys())], 2)
            nnls_dict_list = []
            for area in tqdm(neural_rdms.keys(), leave = False):
                for layer in tqdm(neural_rdms[area].keys(), leave = False):
                    neural_rdm = neural_rdms[area][layer]
                    nnls = kfold_nonnegative_regression(neural_rdm, model_rdm_stack, n_splits=6)
                    nnls_dict_list.append({'model': model_name, 'train_type': train_type,
                                            'area': area, 'layer': layer, 'score': nnls[0]**2, 'method': rsa_subtype})

            pd.DataFrame(nnls_dict_list).to_csv(results_file, index = None)
            
        
        
        

    
    