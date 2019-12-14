#!/usr/bin/env python
# coding: utf-8

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# def conv_block_pre_activation(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
#     '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
#         # Arguments
#             x: input tensor 
#             stage: index for dense block
#             branch: layer index within each dense block
#             nb_filter: number of filters
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#     '''
#     eps = 1.1e-5
#     conv_name_base = 'conv' + str(stage) + '_' + str(branch)
#     relu_name_base = 'relu' + str(stage) + '_' + str(branch)

#     # 1x1 Convolution (Bottleneck layer)
#     inter_channel = nb_filter * 4  
#     x = layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
#     x = layers.Activation('relu', name=relu_name_base+'_x1')(x)
#     x = layers.Convolution2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

#     if dropout_rate:
#         x = layers.Dropout(dropout_rate)(x)

#     # 3x3 Convolution
#     x = layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
#     x = layers.Activation('relu', name=relu_name_base+'_x2')(x)
#     x = layers.ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
#     x = layers.Convolution2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

#     if dropout_rate:
#         x = layers.Dropout(dropout_rate)(x)

#     return x


# def transition_block_with_compression(input_tensor, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
#     ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
#         # Arguments
#             x: input tensor
#             stage: index for dense block
#             nb_filter: number of filters
#             compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#     '''

#     eps = 1.1e-5
#     conv_name_base = 'conv' + str(stage) + '_blk'
#     relu_name_base = 'relu' + str(stage) + '_blk'
#     pool_name_base = 'pool' + str(stage) 

#     x = layers.BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(input_tensor)
#     x = layers.Activation('relu', name=relu_name_base)(x)
#     x = layers.Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, use_bias=False)(x)

#     if dropout_rate:
#         x = layers.Dropout(dropout_rate)(x)

#     x = layers.AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

#     return x

# def dense_block_without_dynamic(input_tensor, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
#     ''' Build a dense_block where the output of each conv_block_pre_activation is fed to subsequent ones
#         # Arguments
#             x: input tensor
#             stage: index for dense block
#             nb_layers: the number of layers of conv_block_pre_activation to append to the model.
#             nb_filter: number of filters
#             growth_rate: growth rate
#             dropout_rate: dropout rate
#             weight_decay: weight decay factor
#             grow_nb_filters: flag to decide to allow number of filters to grow
#     '''

#     eps = 1.1e-5
#     concat_feat = input_tensor

#     for i in range(nb_layers):
#         branch = i+1
#         x = layers.conv_block_pre_activation(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
#         concat_feat = layers.Concatenate()([concat_feat, x])
        
#         if grow_nb_filters:
#             nb_filter += growth_rate

#     return concat_feat, nb_filter

def conv_bn_relu(input_tensor, ch, kernel, activation='post',padding="same", strides=1, weight_decay=5e-4):
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
    
    x = conv_bn_relu(input_tensor, 32, 3, activation , strides=2)
    branch_0 = conv_bn_relu(x, 16, 1, activation)
    branch_0 = conv_bn_relu(branch_0, 32, 3, activation, strides=2)
    
    branch_1 = layers.MaxPooling2D(2)(x)
    
    x = layers.Concatenate()([branch_0, branch_1])
    
    x = conv_bn_relu(x, 32, 1, activation)
    
    return x

def dense_block(input_tensor, num_block, bottleneck_width, k, activation):
    
    x=input_tensor
    k = int(k / 2)
    
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
    
    x=input_tensor
    k = int(k / 2)
    
    for index in range(num_block):
        #left channel     
        inter_channel = int(k*bottleneck_width/4) * 4
        branch_0 = conv_bn_relu(x, inter_channel, 1, activation)
        branch_0 = conv_bn_relu(branch_0, k, 3, activation)
        x = branch_0

    return x

def transition_block(input_tensor, output_channel, activation, is_avgpool=True,compression_factor=1.0):
    
    conv0 = conv_bn_relu(input_tensor, int(output_channel*compression_factor), 1, activation)
    if is_avgpool:
        x= layers.AveragePooling2D(2)(conv0)
    else:
        x=conv0
    
    return x
