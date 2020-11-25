import pandas as pd
import os, sys, torch

import torchvision.models as models

model_types = ['imagenet','inception','segmentation', 'detection', 'video']
pytorch_dirs = dict(zip(model_types, ['.','.','.segmentation.', '.detection.', '.video.']))

path_dir = os.path.dirname(os.path.abspath(__file__))
model_typology = pd.read_csv(path_dir + '/model_typology.csv')
    
training_calls = {'random': '(pretrained=False)', 'pretrained': '(pretrained=True)'}

def define_zoology_options():
    model_options = {}

    for index, row in model_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        for training in ['random', 'pretrained']:
            train_type = row['training'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            model_call = 'models' + pytorch_dirs[model_type] + model_name + training_calls[training]
            model_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                            'train_type': train_type, 'call': model_call})
            
    return model_options

from visual_priors import taskonomy_network

task_typology = pd.read_csv(path_dir + '/task_typology.csv')

def instantiate_taskonomy_model(model_name, verbose = False):
    weights = torch.load(path_dir + '/task_weights/{}_encoder.pth'.format(model_name))
    if verbose: print('{} weights loaded succesfully.'.format(model_name))
    model = taskonomy_network.TaskonomyEncoder()
    model.load_state_dict(weights['state_dict'])
    
    return model

def define_taskonomy_options():
    taskonomy_options = {}

    for index, row in task_typology.iterrows():
        model_name = row['model']
        model_type = 'taskonomy'
        train_type = 'taskonomy'
        model_string = model_name + '_' + train_type
        model_call = "instantiate_taskonomy_model('{}')".format(model_name)
        taskonomy_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                        'train_type': train_type, 'call': model_call})
        
    taskonomy_options['taskonomy_random'] = ({'model_name': 'taskonomy_random', 'model_type': model_type,
                                             'train_type': 'random', 'call': 'taskonomy_network.TaskonomyEncoder()'})
            
    return taskonomy_options

def get_model_options(model_type = None, train_type=None):
    model_options = {**define_zoology_options(), **define_taskonomy_options()}
    if model_type is None and train_type is None :
        return model_options
    if model_type is None and train_type is not None:
        return {string: info for (string, info) in model_options.items() 
                if model_options[string]['train_type'] in train_type}
    if model_type is not None and train_type is None:
        return {string: info for (string, info) in model_options.items() 
                if model_options[string]['model_type'] in model_type}
    if model_type is not None and train_type is not None:
        return {string: info for (string, info) in model_options.items() 
                if model_options[string]['model_type'] in model_type and model_options[string]['train_type'] in train_type}
    
import torchvision.transforms as transforms

image_transforms = {
    'imagenet_from_numpy': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
    'imagenet': transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
    'inception': transforms.Compose([
        transforms.Resize((299,299)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
    'segmentation': transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
    'detection': transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ]),
}
    
def get_image_transforms(model_type = None):
    if model_type is None:
        return image_transforms
    if model_type is not None:
        return image_transforms[model_type]
    
training_printouts = {
    'taskonomy': 'pretrained on taskonomy',
    'imagenet': 'pretrained on imagenet',
    'random': 'randomly initialized'
}

def get_training_printouts(train_type = None):
    if train_type is None:
        return training_printouts
    if train_type is not None:
        return training_printouts[train_type]
