import pandas as pd
import os, sys, torch

import torchvision.models as models

path_dir = os.path.dirname(os.path.abspath(__file__))
model_typology = pd.read_csv(path_dir + '/model_typology.csv')
model_typology['model_name'] = model_typology['model']
model_typology['model_type'] = model_typology['model_type'].str.lower()
    
training_calls = {'random': '(pretrained=False)', 'pretrained': '(pretrained=True)'}

def define_zoology_options():
    model_options = {}
    
    model_types = ['imagenet','inception','segmentation', 'detection', 'video']
    pytorch_dirs = dict(zip(model_types, ['.','.','.segmentation.', '.detection.', '.video.']))

    zoo_typology = model_typology[model_typology['model_type'].isin(model_types)].copy()
    zoo_typology['model_type'] = zoo_typology['model_type'].str.lower()
    for index, row in zoo_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        for training in ['random', 'pretrained']:
            train_type = row['model_type'] if training=='pretrained' else training
            model_string = '_'.join([model_name, train_type])
            model_call = 'models' + pytorch_dirs[model_type] + model_name + training_calls[training]
            model_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                            'train_type': train_type, 'call': model_call})
            
    return model_options

from visual_priors import taskonomy_network

def instantiate_taskonomy_model(model_name, verbose = False):
    weights = torch.load(path_dir + '/task_weights/{}_encoder.pth'.format(model_name))
    if verbose: print('{} weights loaded succesfully.'.format(model_name))
    model = taskonomy_network.TaskonomyEncoder()
    model.load_state_dict(weights['state_dict'])
    
    return model

def define_taskonomy_options():
    taskonomy_options = {}

    task_typology = model_typology[model_typology['train_type'].isin(['taskonomy'])].copy()
    for index, row in task_typology.iterrows():
        model_name = row['model_name']
        model_type = row['model_type']
        train_type = 'taskonomy'
        model_string = model_name + '_' + train_type
        model_call = "instantiate_taskonomy_model('{}')".format(model_name)
        taskonomy_options[model_string] = ({'model_name': model_name, 'model_type': model_type, 
                                        'train_type': train_type, 'call': model_call})
        
    taskonomy_options['random_taskonomy'] = ({'model_name': 'random_taskonomy', 'model_type': 'taskonomy',
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
import torchvision.transforms.functional as functional

def taskonomy_transform(image):
    return (functional.to_tensor(functional.resize(image, 256)) * 2 - 1)#.unsqueeze_(0)

transform_options = {
    'taskonomy': taskonomy_transform,
    'random': [transforms.Resize((224,224)), 
                 transforms.ToTensor()],
    'imagenet': [transforms.Resize((224,224)), 
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])],
    'inception': [transforms.Resize((299,299)), 
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])],
    'detection': [transforms.Resize((224,224)), 
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])],
    'segmentation': 'https://github.com/pytorch/vision/blob/master/references/segmentation/train.py',
    'video': 'https://github.com/pytorch/vision/blob/master/references/video_classification/train.py',
}

def get_transform_options():
    return transform_options
    
def get_recommended_transforms(model_query, input_type = 'PIL'):
    model_types = model_typology['model_type'].unique()
    if model_query in get_model_options():
        model_type = get_model_options()[model_query]['model_type']
    if model_query in model_types:
        model_type = model_query
    if model_query not in list(get_model_options()) + list(model_types):
        raise ValueError('Query is neither a model_string nor a model_type.')
    composable = ['imagenet', 'inception','detection']
    reference = ['segmentation', 'video']
    functionals = ['taskonomy']
    
    if model_type in composable:
        if input_type == 'PIL':
            recommended_transforms = transform_options[model_type]
        if input_type == 'numpy':
            recommended_transforms = [transforms.ToPILImage()] + transform_options[model_type]
        return transforms.Compose(recommended_transforms)
    
    if model_type in functionals:
        if input_type == 'PIL':
            return transform_options[model_type]
        if input_type == 'numpy':
            def functional_from_numpy(image):
                image = functional.to_pil_image(image)
                return transform_options[model_type](image)
            return functional_from_numpy
        
    if model_type in reference:
        recommended_transforms = transform_options[model_type]
        print('Please see {} for best practices.'.format(transform_options))
        
    if model_type not in transform_options:
        print('No reference available for this model class.')
    
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
