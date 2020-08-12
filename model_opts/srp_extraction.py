import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, time, pickle
sys.path.append('..')

import torch as torch
from torch.autograd import Variable

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection

from feature_extraction import *
from model_options import *

def srp_extraction(feature_maps, model_name, output_dir, n_projections=None, n_samples = 119, eps=0.1):
    if n_projections is None:
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)

    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    srp_feature_maps = {}
    for map_key in tqdm(feature_maps):
        feature_map = feature_maps[map_key]
        output_filename = '_'.join([model_name, map_key, 'srp']) + '.npy'
        output_filepath = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_filepath):
            if feature_map.shape[1] >= n_projections:
                srp = SparseRandomProjection(n_projections, random_state=1)
                srp_feature_maps[map_key] = srp.fit_transform(feature_map)
        if not os.path.exists(output_filepath): 
            if feature_map.shape[1] <= n_projections:
                srp_feature_maps[map_key] = feature_map
            np.save(output_filepath, srp_feature_maps[map_key])
        if os.path.exists(output_filepath):
            srp_feature_maps[map_key] = np.load(output_filepath)
            
    return(srp_feature_maps)

if __name__ == "__main__":
    
    torch.cuda.set_device(0)

    brain_data = pickle.load(open('../brain_data/brain_responses_reliable.pkl', 'rb'))
    brain_data = {k1:{k2:v2 for (k2,v2) in brain_data[k1].items() if k2 == 'layer6'} 
                  for (k1,v1) in brain_data.items() if k1 == 'VISpm'}

    mouseland_images = np.load('../stimulus_set.npy')
    image_transforms = get_image_transforms()['imagenet_from_numpy']

    model_options = get_model_options(train_type='random')
    
    for model_string in model_options:
        print('Now attempting {}...', model_name)
        model_name = model_options[model_string]['arch']
        model = model_options[model_string]['call']
        model = model.eval()
        model = model.cuda()
        
        model_inputs = Variable(torch.stack([image_transforms(img) for img in mouseland_images])).cuda()
        mouseland_feature_maps = get_all_feature_maps(model, model_inputs)
        srp_feature_maps = srp_extraction(mouseland_feature_maps, model_name)