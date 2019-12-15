#!/usr/bin/env python
# coding: utf-8

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

'''
All the blocks/layers necessary to build different models are defined as functions in this file
'''


def conv_bn_relu(input_tensor, ch, kernel, activation='post',padding="same", strides=1, weight_decay=5e-4):
    '''
    Depending on activation, convolution block is formed either as post activation or pre activation.    
    input_tensor = input to this block
    ch = Number of channels/features to extract from input_tensor
    kernel = size of convolution kernel
    activation = post for conv-->bn-->relu. else it will be relu-->bn-->conv
    strides = stride of convolution kernel
    weight_decay = regularization coefficient decay factor
    '''
    if activation=='post':
        x = layers.Conv2D(ch, kernel, padding=padding, strides=strides, kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    else:
        x = layers.Activation("relu")(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(ch, kernel, padding=padding, strides=strides, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    return x

def stem(input_tensor,activation):
    '''
    input_tensor = input to this block
    activation = post or pre decides convolution block operation(conv_bn_relu)
    '''
    x = conv_bn_relu(input_tensor, 32, 3, activation , strides=2)
    branch_0 = conv_bn_relu(x, 16, 1, activation)
    branch_0 = conv_bn_relu(branch_0, 32, 3, activation, strides=2)
    
    branch_1 = layers.MaxPooling2D(2)(x)
    
    x = layers.Concatenate()([branch_0, branch_1])
    
    x = conv_bn_relu(x, 32, 1, activation)
    
    return x

def dense_block(input_tensor, num_block, bottleneck_width, k, activation):
    '''
    This is two-way dense block i.e convolution blocks are present in both left and right paths.
    input_tensor = input to this block
    num_block = Number of dense layers in a dense_block
    bottleneck_width = width of bottleneck (dynamic-->changes with each stage or dense block. else it will be constant)
    k = growth rate (decides number of features/channels to extract)
    activation = post or pre decides convolution block operation(conv_bn_relu)
    '''
    x=input_tensor
    k = int(k / 2) #Half of the features are extracted through left and right paths.
    
    for index in range(num_block):
        #left channel     
        inter_channel = int(k*bottleneck_width/4) * 4
        branch_0 = conv_bn_relu(x, inter_channel, 1, activation)
        branch_0 = conv_bn_relu(branch_0, k, 3, activation)
        # right channel
        
        branch_1 = conv_bn_relu(x, inter_channel, 1, activation)
        branch_1 = conv_bn_relu(branch_1, k, 3, activation)
        branch_1 = conv_bn_relu(branch_1, k, 3, activation)

        x = layers.Concatenate()([x, branch_0, branch_1])

    return x

def dense_block_one_way_dynamic(input_tensor, num_block, bottleneck_width, k, activation):
    '''
    This is one-way dense block i.e convolution blocks are present only in one of the left and right paths.
    input_tensor = input to this block
    num_block = Number of dense layers in a dense_block
    bottleneck_width = width of bottleneck (dynamic-->changes with each stage or dense block. else it will be constant)
    k = growth rate (decides number of features/channels to extract)
    activation = post or pre decides convolution block operation(conv_bn_relu)
    '''
    x=input_tensor
    k = int(k / 2)
    
    for index in range(num_block):
        #left channel     
        inter_channel = int(k*bottleneck_width/4) * 4
        branch_0 = conv_bn_relu(x, inter_channel, 1, activation)
        branch_0 = conv_bn_relu(branch_0, k, 3, activation)
        x = layers.Concatenate()([x, branch_0])
    return x

def transition_block(input_tensor, output_channel, activation, is_avgpool=True,compression_factor=1.0):
    '''
    input_tensor = input to this block
    output_channel = Number of features to output to the next stage
    activation = post or pre decides convolution block operation(conv_bn_relu)
    is_avgpool = decide whether to use average pooling after 1*1 convolution
    compression_factor = reduce number of features to be extracted using this factor(<=1)
    '''
    conv0 = conv_bn_relu(input_tensor, int(output_channel*compression_factor), 1, activation)
    if is_avgpool:
        x= layers.AveragePooling2D(2)(conv0)
    else:
        x=conv0
    return x
