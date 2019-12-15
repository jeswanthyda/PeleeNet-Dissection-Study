from utils.layer_utils import *

def PeleeNet(input_shape=(224,224,3),use_stem_block=True, k=32, block_config=[3,4,8,6], out_layers = [128,256,512,704],bottleneck_width=[1,2,4,4],n_classes=198, activation='post'):

    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def model5(input_shape=(224,224,3),use_stem_block=True, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[1,2,4,4],n_classes=198, activation='post'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def model4(input_shape=(224,224,3),use_stem_block=True, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[1,2,4,4],n_classes=198, activation='pre'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block_one_way_dynamic(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def model3(input_shape=(224,224,3),use_stem_block=False, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[1,2,4,4],n_classes=198, activation='pre'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block_one_way_dynamic(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def model2(input_shape=(224,224,3),use_stem_block=False, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[1,1,1,1],n_classes=198, activation='post'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block_one_way_dynamic(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def model1(input_shape=(224,224,3),use_stem_block=False, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[2,2,2,2],n_classes=198, activation='pre'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block_one_way_dynamic(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model

def DenseNet41(input_shape=(224,224,3),use_stem_block=False, k=32, block_config=[3,4,8,3], out_layers = [128,256,512,704],bottleneck_width=[4,4,4,4],n_classes=198, activation='pre'):
    
    inputs = layers.Input(shape=input_shape)
    x=stem(inputs, activation) if use_stem_block else inputs
    
    for i in range(4):
        
        x = dense_block_one_way_dynamic(x, block_config[i], bottleneck_width[i], k , activation)
        use_pooling = i < 3
        x = transition_block(x, out_layers[i], activation, is_avgpool=use_pooling,compression_factor=0.5)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(n_classes, activation="softmax")(x)
    
    model = keras.models.Model(inputs, x, name="PeleeNet")
    
    return model