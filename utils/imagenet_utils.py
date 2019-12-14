#!/usr/bin/env python
# coding: utf-8

import _pickle as pickle
import os
import tarfile
import glob
import urllib.request as url
import numpy as np
import split_folders
import numpy as np
from PIL import Image

def download_data():
    """
    Download the CIFAR-10 data from the website, which is approximately 170MB.
    The data (a .tar.gz file) will be store in the ./data/ folder.
    :return: None
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
    Unpack the CIFAR-10 dataset and load the datasets.
    :param mode: 'train', or 'test', or 'all'. Specify the training set or test set, or load all the data.
    :return: A tuple of data/labels, depending on the chosen mode. If 'train', return training data and labels;
    If 'test' ,return test data and labels; If 'all', return both training and test sets.
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/imagenet_images'):
        download_data()
    # Split data folder into test train and valid folders
    if not os.path.exists('./split_data'):    
        split_folders.ratio('./data/imagenet_images', output="./split_data", seed=1, ratio=(.8, .2))
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    
    #Load training data    
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
            
    
    
    
    
    
    
    
    