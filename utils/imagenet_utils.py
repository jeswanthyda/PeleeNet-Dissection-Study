#!/usr/bin/env python
# coding: utf-8

import os
import split_folders
import numpy as np
from PIL import Image

def download_data():
    """
    Download subset of imagenet dataset into './data' with almost 200 classes and around 500 images in each class
    by scraping through URLs provided at http://image-net.org/. Work by Martins Frolovs 
    (https://github.com/mf1024/ImageNet-Datasets-Downloader) has been used  and can be found in imagenet_download folder.
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('Start downloading data...')
        get_ipython().run_line_magic('run', "-i './imagenet_download/downloader.py'")
        print('Download complete.')
    else:
        if os.path.exists('./data'):
            print('Imagenet already exists.')
            
            

          
            
            
def load_data():
    """
    Check if data has been downloaded and split it into train and val folders in './split_data' in 8:2 ratio.
    Read each image in RGB format and resize it to (224,224,3) which is compatible to feed as input to network.
    Returns train_data,train_label,val_data,val_label as numpy arrays.
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/imagenet_images'):
        download_data()
    # Split data folder into train and valid folders
    if not os.path.exists('./split_data'):    
        split_folders.ratio('./data/imagenet_images', output="./split_data", seed=1, ratio=(.8, .2))
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    
    #Load training data and process to required format   
    val_dir = os.path.join(os.getcwd(),'split_data/val')
    
    train_dir = os.path.join(os.getcwd(),'split_data/train')

    for label in os.listdir(train_dir):
        for sample_dir in os.listdir(os.path.join(train_dir,label)):
            image = Image.open(os.path.join(train_dir,label,sample_dir))
            image = image.convert('RGB')
            image = np.array(image.resize((224,224)))
            train_data.append(image)
            train_label.append(label)
            
            
    for label in os.listdir(val_dir):
        for sample_dir in os.listdir(os.path.join(val_dir,label)):
            image = Image.open(os.path.join(val_dir,label,sample_dir))
            image = image.convert('RGB')
            image = np.array(image.resize((224,224)))
            val_data.append(image)
            val_label.append(label)
            
            
            
    return train_data,train_label,val_data,val_label
            
    
    
    
    
    
    
    
    
