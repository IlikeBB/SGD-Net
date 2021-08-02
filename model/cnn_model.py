import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
act = 'relu'
# def get_model(depth=32, width=128, height=128, class_num=3, classification_layer='sigmoid'):
#     """Build a 3D convolutional neural network model."""

#     inputs = keras.Input((depth, width, height, 1))

#     cx1 = layers.Conv3D(filters=64, kernel_size=3)(inputs)
#     ax1 = layers.Activation(act)(cx1)
#     px1 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax1)
#     bx1 = layers.BatchNormalization()(px1)
#     print(bx1.shape)
    
#     cx2 = layers.Conv3D(filters=64, kernel_size=3)(bx1)
#     ax2 = layers.Activation(act)(cx2)
#     px2 = layers.MaxPool3D(pool_size=(1, 2, 2))(ax2)
#     bx2 = layers.BatchNormalization()(px2)
#     print(bx2.shape)
    
#     cx4 = layers.Conv3D(filters=128, kernel_size=3)(bx2)
#     ax4 = layers.Activation(act)(cx4)
#     px4 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax4)
#     bx4 = layers.BatchNormalization()(px4)
#     print(bx4.shape)
    
#     cx5 = layers.Conv3D(filters=256, kernel_size=3)(bx4)
#     ax5 = layers.Activation(act)(cx5)
#     px5 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax5)
#     bx5 = layers.BatchNormalization()(px5)
#     print(bx5.shape)

#     x = layers.GlobalAveragePooling3D()(bx5)

#     x = layers.Dense(units=512)(x)
#     x = layers.Activation(act)(x)
#     x = layers.Dropout(0.4)(x)

#     x = layers.Dense(units=256)(x)
#     x = layers.Activation(act)(x)
#     x = layers.Dropout(0.4)(x)
    
#     outputs = layers.Dense(units=class_num, activation=classification_layer, name='classification')(x)

#     # Define the model.
# #     model = keras.Model(inputs, [outputs, outputs2], name="3dcnn")
#     model = keras.Model(inputs, outputs, name="3dcnn")
    
#     return model

def get_model(depth=32, width=384, height=384, class_num=1, classification_layer='sigmoid'):
    inputs = keras.Input((depth, width, height, 1))
    
    conv_layer1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(inputs)
    pooling_layer1 = MaxPool3D(pool_size=(2, 6, 6))(conv_layer1)
    pooling_layer1 = BatchNormalization()(pooling_layer1)  
    # print(pooling_layer1.shape)
    conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    pooling_layer2 = MaxPool3D(pool_size=(2, 4, 4))(conv_layer2)
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    # print(pooling_layer2.shape)
    conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer3)
    pooling_layer3 = BatchNormalization()(pooling_layer3)
    # print(pooling_layer3.shape)
    conv_layer4 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu')(pooling_layer3)
    pooling_layer4 = MaxPool3D(pool_size=(1, 2, 2))(conv_layer4)
    pooling_layer4 = BatchNormalization()(pooling_layer4)
    # print(pooling_layer4.shape)
    conv_layer5 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(pooling_layer4)
    pooling_layer5 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer5)
    # print(pooling_layer5.shape)
    pooling_layer9 = BatchNormalization()(pooling_layer5)
    flatten_layer = Flatten()(pooling_layer9)
    
    dense_layer3 = Dense(units=512, activation='relu')(flatten_layer)
    dense_layer3 = Dropout(0.4)(dense_layer3)

    dense_layer4 = Dense(units=256, activation='relu')(dense_layer3)
    dense_layer4 = Dropout(0.4)(dense_layer3)
  
    output_layer = Dense(units=1, activation=classification_layer)(dense_layer4)

    model = Model(inputs=inputs, outputs=output_layer)
    return model


def __main__(): 
    print('pass')
    model = get_model()
    model.summary()