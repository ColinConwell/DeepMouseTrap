import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, argparse, pickle

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

sys.path.append('../neural_data')
from neural_data import *

sys.path.append('../model_opts')
from feature_extraction import *
from feature_reduction import *
from model_options import *

from neural_regression import *

# MouseLand Sparse Random Projection + Ridge Regression for PyTorch Zoo

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sparse Random Projection + Ridge Regression')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--srp_eps', type=float, required=False, default = 0.1,
                        help='quality of embeddings to extract with sparse random projection')
    parser.add_argument('--srp_seed', type=int, required=False, default = 0,
                        help='random seed for random projections')
    parser.add_argument('--srp_out_dir', type=str, required=False, default='./srp_arrays',
                        help='directory for output from random projections')
    parser.add_argument('--results_dir', type=str, required=False, default='./srp_ridge_results',
                        help='(sub) directory for results')
    parser.add_argument('--cuda_device', type=int, required=False, default=0,
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    model_string = args.model_string
    srp_eps = args.srp_eps
    srp_seed = args.srp_seed
    srp_out_dir = args.srp_out_dir
    results_dir = args.results_dir
    cuda_device = args.cuda_device
    
    torch.cuda.set_device(1)
    
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
        
    neural_data = AllenBrainObservatory()
    stimulus_set = neural_data.stimulus_set
    cell_responses = neural_data.cell_responses
    #cell_response_split1 = pd.read_csv('../neural_data/brain_observatory/cell_response_even.csv').set_index('cell_specimen_id')
    #cell_response_split2 = pd.read_csv('../neural_data/brain_observatory/cell_response_odd.csv').set_index('cell_specimen_id')
    

    image_transforms = get_recommended_transforms(model_string, input_type = 'numpy')
    model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set]))
    srp_feature_maps = srp_extraction(model_string, model_inputs = model_inputs, output_dir = srp_out_dir, seed = srp_seed)
    
    n_projections = next(iter(srp_feature_maps.values())).shape[1]
    
    print('Regressing {} random projections from {} {}; saving results into {}...'
          .format(n_projections, model_name, get_training_printouts(train_type), results_dir))
    
    method_desc = 'srp_ridge_regression_{}_{}_with_split_half'.format(n_projections, srp_seed)
    
    results_dir = os.path.join(results_dir, str(n_projections) + '_' + str(srp_seed), model_string)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for model_layer in tqdm(srp_feature_maps):
        results_filepath = os.path.join(results_dir, model_layer + '.csv')
            
        if not os.path.exists(results_filepath):
            feature_map = srp_feature_maps[model_layer]
            X, y = scale(feature_map), cell_responses.to_numpy().transpose()
            scores = neural_regression(X, y, score_type = ['pearson_r', 'explained_variance'], cv_splits = 'gcv')
        
            scoresheet_list = []
            for score_type in scores:
                scoresheet_list.append(pd.DataFrame({'cell_specimen_id': cell_responses.index,
                                                     'score_type': score_type, 'score': scores[score_type],
                                                     'model': model_name, 'train_type': train_type, 
                                                     'model_layer': model_layer, 'method': method_desc}))
                    
                
            scoresheet = pd.concat(scoresheet_list)
            scoresheet.to_csv(results_filepath, index = None)
               
            
        