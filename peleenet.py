#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# In[16]:


import pandas as pd
import os
from tensorflow.keras.layers import Input,Dense,Activation,GlobalAveragePooling2D,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization,AveragePooling2D,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers


def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1),weight_decay=5e-4):
    
    x = Conv2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    
    return x


def stem(x,num_init_channel=32):
    
    conv1_output = conv2d_bn(x, num_init_channel, 3, 3,strides=(2,2))
    branch_0 = conv2d_bn(conv1_output, int(num_init_channel/2), 1, 1)
    branch_0 = conv2d_bn(branch_0, num_init_channel, 3, 3,strides=(2,2))
    #print("the left branch is calculated")
    branch_1 = MaxPooling2D(2, strides=(2,2))(conv1_output)
    #print("the right branch is calulated")
    out = Concatenate()([branch_0, branch_1])
    output = conv2d_bn(out, num_init_channel, 1, 1)
    
    return output

def dense_block(x, num_block, bottleneck_width, k=32):
    
    inter_channel = k*bottleneck_width
    output=x
    
    for index in range(num_block):
        #left channel     
        conv_branch_0 = conv2d_bn(output, inter_channel, 1, 1)
        conv_branch_0 = conv2d_bn(conv_branch_0, k, 3, 3)
        # right channel
        conv_branch_1 = conv2d_bn(output, inter_channel, 1, 1)
        conv_branch_1 = conv2d_bn(conv_branch_1, k, 3, 3)
        conv_branch_1 = conv2d_bn(conv_branch_1, k, 3, 3)

        output = Concatenate()([output, conv_branch_0, conv_branch_1])

    return output


def transition_block(x, output_channel, is_avgpool=True):
    
    conv0 = conv2d_bn(x, output_channel, 1, 1, strides=(2,2))
    if is_avgpool:
        output=AveragePooling2D((2,2),strides=(2,2))(conv0)
    else:
        output=conv0
    return output


def classification_layer(x, n_classes=1000):
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation="softmax")(x)
    return x


def PeleeNet(input_shape=(224,224,3),use_stem_block=True, num_init_channel=3, k=32, block_config=[3,4,8,6], out_layers = [128,256,512,704],bottleneck_width=[2,2,4,4],n_classes=1000):
    
    inputs = Input(shape=input_shape)
    x=stem(inputs,num_init_channel) if use_stem_block else inputs
    for i in range(4):
        x = dense_block(x,block_config[i], bottleneck_width[i],k)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], is_avgpool=use_pooling)
    x=classification_layer(x, n_classes) 
    
    model = Model(inputs, x, name="peleenet")
    
    return model
