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

parser = argparse.ArgumentParser(description="Compute the RDM for a target model.")
parser.add_argument('--model_string', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')

if __name__ == "__main__":
    
    args = parser.parse_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    model_string = args.model_string
    model_options = get_model_options()
    model_name = model_options[model_string]['model_name']
    model_type = model_options[model_string]['model_type']
    model_call = model_options[model_string]['call']
    image_transforms = get_image_transforms()['imagenet_from_numpy']
    output_file = os.path.join(output_dir, model_string + '_rdms.pkl')
    
    if not os.path.exists(output_file):
    
        print('Computing RDMS for {} on {}...'.format(model_string, torch.cuda.get_device_name(0)))
        torch.cuda.set_device(0)
        model = eval(model_call)
        model = model.eval()
        model = model.cuda()
     
        stimulus_set = np.load('../stimulus_set.npy')
        
        model_inputs = Variable(torch.stack([image_transforms(img) for img in stimulus_set])).cuda()
        feature_maps = get_all_feature_maps(model, model_inputs)
        
        model_rdms = {}
        for model_layer in tqdm(feature_maps):
            model_rdms[model_layer] = np.corrcoef(feature_maps[model_layer])

        with open(output_file, 'wb') as file:
            pickle.dump(model_rdms, file)
        

    
    