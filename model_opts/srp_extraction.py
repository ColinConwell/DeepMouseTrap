import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, time, pickle, argparse
sys.path.append('..')

import torch as torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

from feature_extraction import *
from model_options import *

def get_feature_map_filepaths(model_string, feature_map_names, output_dir, subtype='srp'):
    return {feature_map_name: os.path.join(output_dir, '_'.join([model_string, feature_map_name, subtype]) + '.npy')
                                for feature_map_name in feature_map_names}
    

def srp_extraction(model_string, feature_maps = None, model_inputs=None, output_dir='./srp_arrays', 
                   n_projections=None, eps=0.1, seed = 1, device = 0):
    
    if feature_maps == None and model_inputs == None:
        raise ValueError('Neither feature_maps nor model_inputs are defined.')
        
    if model_inputs is not None and not isinstance(model_inputs, (DataLoader, torch.Tensor)):
        raise ValueError('model_inputs not supplied in recognizable format.')
    
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    device_name = 'CPU' if device is None else torch.cuda.get_device_name(device)
    
    if n_projections is None:
        if feature_maps is None:
            if isinstance(model_inputs, torch.Tensor):
                n_samples = len(model_inputs)
            if isinstance(model_inputs, DataLoader):
                n_samples = len(model_inputs.dataset)
        if feature_maps is not None:
            n_samples = next(iter(feature_maps.items())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    print('Computing {} SRPs for {} on {}...'.format(n_projections, model_string, device_name))

    output_dir = os.path.join(output_dir, str(n_projections) + '_' + str(seed), model_string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    srp_feature_maps = {}
    if feature_maps is None:
        model = eval(model_call)
        model = model.eval()
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            model = model.cuda()
        
        feature_map_names = get_empty_feature_maps(model, names_only = True)
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
        if not all([os.path.exists(file) for file in output_filepaths.values()]):
            feature_maps = get_all_feature_maps(model, model_inputs)
            
    if feature_maps is not None:
        feature_map_names = list(feature_maps.keys())
        output_filepaths = get_feature_map_filepaths(model_string, feature_map_names, output_dir)
        
    for feature_map_name in tqdm(feature_map_names):
        output_filepath = output_filepaths[feature_map_name]
        if not os.path.exists(output_filepath):
            feature_map = feature_maps[feature_map_name]
            if feature_map.shape[1] >= n_projections:
                srp = SparseRandomProjection(n_projections, random_state=seed)
                srp_feature_maps[feature_map_name] = srp.fit_transform(feature_map) 
            if feature_map.shape[1] <= n_projections:
                srp_feature_maps[feature_map_name] = feature_map
            np.save(output_filepath, srp_feature_maps[feature_map_name])
        if os.path.exists(output_filepath):
            srp_feature_maps[feature_map_name] = np.load(output_filepath)
            
    return(srp_feature_maps)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute SRPs for a target model.")
    parser.add_argument('--model_string', type=str, required=True)
    parser.add_argument('--n_projections', type=int, required=False, default=None)
    parser.add_argument('--output_dir', type=str, required=False, default='srp_arrays')
    parser.add_argument('--cuda_device', type=int, required=False, default=0)
    
    args = parser.parse_args()
    model_string = args.model_string
    output_dir = args.output_dir
    n_projections = args.n_projections
    cuda_device = args.cuda_device
    
    stimulus_set = np.load('../stimulus_set.npy')
    image_transforms = get_image_transforms()['imagenet_from_numpy']
    model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set]))
    srp_feature_maps = srp_extraction(model_string, model_inputs = model_inputs, output_dir = output_dir, n_projections = n_projections)