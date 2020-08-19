from fastai.torch_core import flatten_model
import torchvision.models as models
from tqdm import tqdm as tqdm
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import pickle
import os, sys
import torch

sys.path.append('../model_opts')
from feature_extraction import *
from model_options import *

def rdm_extraction(model_string, feature_maps = None, model_inputs = None, output_dir='rdm_arrays', device = 0):
    
    if feature_maps == None and model_inputs == None:
        print('Neither feature maps nor model_inputs are defined.')
    
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    device_name = 'CPU' if device is None else torch.cuda.get_device_name(device)
    print('Computing RDMS for {} on {}...'.format(model_string, device_name))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, model_string + '_rdms.pkl')
    if os.path.exists(output_file):
        model_rdms = pickle.load(open(output_file,'rb'))
        
    if not os.path.exists(output_file):
        model = eval(model_call)
        model = model.eval()
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            model = model.cuda()
        
        if feature_maps is None:
            feature_maps = get_all_feature_maps(model, model_inputs)
        
        model_rdms = {}
        for model_layer in tqdm(feature_maps):
            model_rdms[model_layer] = np.corrcoef(feature_maps[model_layer])
        with open(output_file, 'wb') as file:
            pickle.dump(model_rdms, file)
            
    return(model_rdms)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute the RDM for a target model.")
    parser.add_argument('--model_string', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=False, default='rdm_arrays')
    parser.add_argument('--cuda_device', type=int, required=False, default=0)
    
    args = parser.parse_args()
    model_string = args.model_string
    output_dir = args.output_dir
    cuda_device = args.cuda_device
    
    stimulus_set = np.load('../stimulus_set.npy')
    image_transforms = get_image_transforms()['imagenet_from_numpy']
    model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set]))
    model_rdms = rdm_extraction(model_string, model_inputs = model_inputs, output_dir = output_dir)
        

    
    