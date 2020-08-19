import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from collections import OrderedDict
from tqdm.auto import tqdm as tqdm
from PIL import Image
import torch.nn as nn
import torch, torchvision
import pandas as pd
import numpy as np

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
}

def get_image_transforms():
    return(image_transforms)

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)


# Method 1: Flatten model; extract features by layer

from fastai.torch_core import flatten_model

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.out = output.clone().detach().requires_grad_(True).cuda()
    def close(self):
        self.hook.remove()
    def extract(self):
        return self.out
    
def get_layer_names(layers):
    layer_names = []
    for layer in layers:
        layer_name = str(layer).split('(')[0]
        layer_names.append(layer_name + '-' + str(sum(layer_name in string for string in layer_names) + 1))
    return layer_names

def get_features_by_layer(model, target_layer, img_tensor):
    features = SaveFeatures(target_layer)
    model(img_tensor)
    features.close()
    return features.extract()


# Method 2: Hook all layers simultaneously; remove duplicates

def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in key for key in list(module_list.keys())) + 1)
    
    return '-'.join([class_name, class_count])
    
def get_layer_names(model, output='dict'):
    layer_name_list = []
    layer_name_dict = OrderedDict()
    def add_layer_to_list(module):
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            layer_key = get_module_name(module, layer_name_dict)
            layer_name_dict[layer_key] = None
            layer_name_list.append(layer_key)
            
    model.apply(add_layer_to_list)
            
    if output=='list':
        return layer_name_list
    if output=='dict':
        return layer_name_dict
    if output=='both':
        return layer_name_list, layer_name_dict
    
def get_feature_map_count(model):
    module_list = []
    def count_module(module):
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            module_list.append(module)
            
    model.apply(count_module)
            
    return len(module_list)
    
def remove_duplicate_feature_maps(feature_maps, return_matches = False):
    matches = []
    layer_names = list(feature_maps.keys())
    for i in range(len(layer_names)):
        for j in range(i+1,len(layer_names)):
            layer1 = feature_maps[layer_names[i]].flatten()
            layer2 = feature_maps[layer_names[j]].flatten()
            if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                if layer_names[j] not in matches:
                    matches.append(layer_names[j])

    for match in matches:
        feature_maps.pop(match)
    
    if return_matches:
        return(feature_maps, matches)
    
    if not return_matches:
        return(feature_maps)
    
def get_feature_maps(model, inputs, layers_to_retain = None, use_tqdm = True, remove_duplicates = True):
    def register_hook(module):
        def hook(module, input, output):
            module_name = get_module_name(module, feature_maps)
            feature_maps[module_name] = output.cpu().detach()
                
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
            
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
        
    if remove_duplicates == True:
        remove_duplicate_feature_maps(feature_maps)
        
    if layers_to_retain is not None:
        feature_maps = {map_key: map_item  for (map_key, map_item) in feature_maps.items() 
                            if map_key in layers_to_retain}
    
    return(feature_maps)

def get_empty_feature_maps(model, input_size=(3,224,224), dataset_size=1, layers_to_retain = None, 
        remove_duplicates = True, names_only=False):
    inputs = torch.rand(1, *input_size).type(torch.FloatTensor)
    inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
    empty_feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
    
    for map_key in empty_feature_maps:
        empty_feature_maps[map_key] = torch.empty(dataset_size, *empty_feature_maps[map_key].shape[1:])
        
    if names_only == True:
        return list(empty_feature_maps.keys())
    
    if names_only == False:
        return empty_feature_maps  

def get_all_feature_maps(model, inputs, layers_to_retain=None, use_tqdm = True, flatten=True, numpy=True):
    if isinstance(inputs, DataLoader):
        input_size, dataset_size, start_index = inputs.dataset[0].shape, len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model, input_size, dataset_size, layers_to_retain)
        for i, imgs in enumerate(tqdm(inputs)) if use_tqdm else enumerate(inputs):
            imgs = imgs.cuda() if next(model.parameters()).is_cuda else imgs
            batch_feature_map = get_feature_maps(model, imgs)
            for layer in feature_maps.keys():
                feature_maps[layer][start_index:start_index+imgs.shape[0],...] = batch_feature_map[layer]
            start_index += imgs.shape[0]
                    
    if not isinstance(inputs, DataLoader):
        inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs, layers_to_retain)
        
    feature_maps = remove_duplicate_feature_maps(feature_maps)
    
    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)
            
    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy()
            
    return feature_maps

def get_all_feature_maps_bystack(model, inputs, layers_to_retain=None, use_tqdm = True, flatten=True, numpy=True):
    if isinstance(inputs, DataLoader):
        if layers_to_retain is None:
            feature_maps = get_empty_feature_maps(model)
        if layers_to_retain is not None:
            feature_maps = {layer: None for layer in layers_to_retain}
        for i, imgs in enumerate(tqdm(inputs)) if use_tqdm else enumerate(inputs):
            imgs = imgs.cuda() if next(model.parameters()).is_cuda else imgs
            batch_feature_map = get_feature_maps(model, imgs)
            for layer in feature_maps.keys():
                if feature_maps[layer] == None:
                    feature_maps[layer] = batch_feature_map[layer]
                if feature_maps[layer] != None:
                    feature_maps[layer] = torch.cat((feature_maps[layer], batch_feature_map[layer]))
    
    if not isinstance(inputs, DataLoader):
        inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs)
        
    feature_maps = remove_duplicate_feature_maps(feature_maps)
    
    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)
            
    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy()
            
    return feature_maps
        
# Helpers: Dataloaders and functions for facilitating feature extraction
        
class Array2DataLoader(Dataset):
    def __init__(self, img_array, image_transforms=None):
        self.transforms = image_transforms
        if isinstance(img_array, np.ndarray):
            self.images = img_array
        if isinstance(img_array, str):
            self.images = np.load(img_array)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.images.shape[0]
    
def get_feature_map_size(feature_maps, layer=None):
    total_size = 0
    if layer is None:
        for map_key in feature_maps:
            if isinstance(feature_maps[map_key], np.ndarray):
                total_size += feature_maps[map_key].nbytes / 1000000
            elif torch.is_tensor(feature_maps[map_key]):
                total_size += feature_maps[map_key].numpy().nbytes / 1000000
        return total_size
    
    if layer is not None:
        if isinstance(feature_maps, np.ndarray):
            return feature_maps[layer].nbytes / 1000000
        elif torch.is_tensor(feature_maps):
            return feature_maps[layer].nbytes / 1000000
        
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        