import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import os, sys, argparse, pickle

sys.path.append('../model_opts')
from feature_extraction import *
from model_options import *
from srp_extraction import *
from neural_regression import *

# Ridge Regression

def anscombe_transform(x):
    return 2.0*np.sqrt(x + 3.0/8.0)

def kfold_ridge_regression(X, y, n_splits = 6):
    kfolds = KFold(n_splits, random_state = 1)
    y = anscombe_transform(y)
    pred_y = np.zeros(119)
    for train_indices, test_indices in kfolds.split(np.arange(119)):
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = Ridge(alpha=1.0).fit(train_X, train_y)
        pred_y[test_indices] = regression.predict(test_X)
    return(pearsonr(pred_y.squeeze(), y)[0]**2)

brain_data_dir = '../neural_data/'

# MouseLand Sparse Random Projection + Ridge Regression for PyTorch Zoo

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sparse Random Projection + Ridge Regression')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--n_projections', type=int, required=True,
                        help='number of components to extract with sparse random projection')
    parser.add_argument('--output_dir', type=str, required=False, default='./',
                        help='(root) directory for all outputs')
    parser.add_argument('--results_dir', type=str, required=False, default='srp_ridge_halfdata_results',
                        help='(sub) directory for results')
    parser.add_argument('--cuda_device', type=int, required=False, default=0,
                        help='target cuda device for gpu compute')
    
    args = parser.parse_args()
    
    model_string = args.model_string
    n_projections = args.n_projections
    output_rootdir = args.output_dir
    results_dir = args.results_dir
    
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    train_type = model_options[model_string]['train_type']
    model_call = model_options[model_string]['call']
    image_transforms = get_image_transforms()['imagenet_from_numpy']
        
    print('Regressing {} random projections from {} {} with {}; saving results into {}...'
          .format(n_projections, model_name, get_training_printouts(train_type), 
                  torch.cuda.get_device_name(args.cuda_device), output_rootdir))
    
    torch.cuda.set_device(args.cuda_device)
    
    stimulus_set = np.load('../stimulus_set.npy')
    stimulus_set 
    
    cell_responses = pd.read_csv(brain_data_dir + 'cell_response_average.csv').set_index('cell_specimen_id')
    cell_data = pd.read_csv(brain_data_dir + 'cell_data_plus.csv')
    cell_subset = cell_data[((cell_data['p_run_mod_ns'].isna()) | 
                            (cell_data['p_run_mod_ns'] > 0.05)) & 
                            (cell_data['p_ns'] < 0.05) &
                            (cell_data['splithalf_r'] > 0.75)]['cell_specimen_id'].to_list()
    cell_response_subset = cell_responses.loc[cell_subset,:]
    
    model = eval(model_call)
    model = model.eval()
    model = model.cuda()
    
    model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set])).cuda()
    feature_maps = get_all_feature_maps(model, model_inputs)
    
    output_dir = os.path.join(output_rootdir, 'srp_arrays_halfdata', str(n_projections), train_type)
    feature_maps_reduced = srp_extraction(feature_maps, model_name, output_dir, n_projections)
    
    output_dir = os.path.join(output_rootdir, results_dir, model_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_layer_name in tqdm(feature_maps_reduced):
        output_filename = '_'.join([model_layer_name])
        output_filepath = os.path.join(output_dir, model_layer_name + '.csv')
        if not os.path.exists(output_filepath):
            scoresheet_dictlist = []
            for cell in tqdm(cell_subset, leave=False):
                feature_map = feature_maps_reduced[model_layer_name]
                score = kfold_ridge_regression(feature_map, cell_response_subset.loc[cell,:].to_numpy())
                scoresheet_dictlist.append({'cell_specimen_id': cell, 'score': score, 'model': model_name, 'train_type': train_type,
                                   'model_layer': model_layer_name, 'method': 'srp_ridge_{}_projections'.format(n_projections)})
            pd.DataFrame(scoresheet_dictlist).to_csv(output_filepath, index = None)
