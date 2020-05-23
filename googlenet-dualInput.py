
import os
import numpy as np
import tensorflow as tf

from skimage.io import imread
from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

import pickle

# define parameters
CLASS_NUM = 3
BATCH_SIZE = 128
EPOCH_STEPS = int(1800/BATCH_SIZE)
IMAGE_SHAPE = (224, 224, 3)
IMAGE_TRAIN = '/home/smartyy-pants/Documents/Ceeri/CNN-Threat-Detection/ImagesVladmir/Images/'
MODEL_NAME = 'googlenet_dualInput_threat.h5'

model = None

as_gray = True
in_channel = 4

if as_gray:
  in_channel = 1

def read_images(file_paths, as_gray, channels):
  images = []
  for file_path in file_paths:
    images.append(imread(file_path, as_gray = as_gray))
    
  images = np.asarray(images, dtype=np.float32)
  # normalize
  images = images / 255.0 #np.max(images)
  # reshape to match Keras expectaions
  images = images.reshape(images.shape[0], IMAGE_SHAPE[0], IMAGE_SHAPE[1], channels)
  return images

# prepare data
def getData():
    data = []   # n*3 format
    folders = os.listdir(IMAGE_TRAIN)
    for folder in folders:
        print("Reading Images in ",os.path.join(IMAGE_TRAIN,folder,""))
        all_pairs = os.listdir(os.path.join(IMAGE_TRAIN,folder,""))
        for pairs in all_pairs:
            images = os.listdir(os.path.join(IMAGE_TRAIN,folder,pairs,""))
            pair_data = []
            for image in images:
                pair_data.append(os.path.join(IMAGE_TRAIN,folder,pairs,image,""))
                
            label = [int(i) for i in list(folder)] 
            pair_data.append(label)
            data.append(pair_data)
            
    np.random.shuffle(data)
    return data
    
# create model
def inception(x, filters):
    # 1x1
    path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
    
    # 1x1->5x5
    path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)

    return Concatenate(axis=-1)([path1,path2,path3,path4])


def auxiliary(x, name=None):
    layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
    layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu',name=name)(layer)
    return layer

def combineAuxiliary(layer1,layer2,name=None):
    layer = Concatenate()([layer1, layer2])
    layer = Flatten()(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units=CLASS_NUM, activation='sigmoid', name=name)(layer)
    return layer

def singleInputModel(name=None):
    layer_in = Input(shape=IMAGE_SHAPE)
    
    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    # stage-3
    layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
    layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-4
    layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
    aux1  = auxiliary(layer, name=name+'-aux1')
    layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
    layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
    layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
    aux2  = auxiliary(layer, name=name+'-aux2')
    layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-5
    layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
    layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
    layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid',name=name)(layer)
    
    return [layer_in,layer,aux1,aux2]

def dualInputModel():
    
    network1 = singleInputModel('network1')
    network2 = singleInputModel('network2')
    
    input_1 = network1[0]
    input_2 = network2[0]

    layer = Concatenate()([network1[1],network2[1]])

    layer = Flatten()(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units=256, activation='linear')(layer)
    
    main = Dense(units=CLASS_NUM, activation='sigmoid', name='main')(layer)
    
    aux1_final = combineAuxiliary(network1[2],network2[2],'aux1_final')
    aux2_final = combineAuxiliary(network1[3],network2[3],'aux2_final')
    
    model = Model(inputs=[input_1,input_2], outputs=[main, aux1_final, aux2_final])
    
    return model

data = []

if os.path.isfile('Images_path.pkl'): 
    with open(r"Images_path.pkl", "rb") as input_file:
        data = pickle.load(input_file)
else:
    data = getData()
    with open(r"Images_path.pkl", "wb") as output_file:
        pickle.dump(data, output_file)
    
data = np.array(data)
x1_loc, x2_loc, y = data.T
x1 = read_images(x1_loc,as_gray,in_channel)
x2 = read_images(x2_loc,as_gray,in_channel)

x_train_comp = np.stack((x1, x2), axis=4)

x_train, x_test, y_train, y_test = train_test_split(x_train_comp, y, test_size = 0.3, random_state=666)

# take them apart
x1_train = x_train[:,:,:,:,0]
x1_test = x_test[:,:,:,:,0]

x2_train = x_train[:,:,:,:,1]
x2_test = x_test[:,:,:,:,1]


def train():
        # data  :   1800*3

        optimizer = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # optimizer = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        # epochs = [20, 30, 20, 30]
        epochs=[5,5,5,5]
        history_all = {}

        checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        callbacks = [checkpoint]

        model.fit([x1_train, x2_train], y_train,
                batch_size=BATCH_SIZE,
                epochs=5,
                callbacks=callbacks,
                verbose=1,
                validation_data=([x1_test, x2_test], y_test),
                shuffle=True)

def test():
    # load weights
    model.load_weights(MODEL_NAME)
    final_loss, final_acc = model.evaluate([x1_test, x2_test], y_test, verbose=1)
    print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))     

def __init__():
    model = dualInputModel()
    # model.summary()
    # model.load_weights(MODEL_NAME)
    # tf.keras.utils.plot_model(model, 'GoogLeNet.png')
    train()
    # test()
    
    
#     # save history    
#     if len(history_all) == 0:
#         history_all = {key: [] for key in train_history.history}
    
#     for key in history_all:
#         history_all[key].extend(train_history.history[key])

# model.save(MODEL_NAME)

# # show train history
# def show_train_history(history, xlabel, ylabel, train):
#     for item in train:
#         plt.plot(history[item])
#     plt.title('Train History')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(train, loc='upper left')
#     plt.show()

# show_train_history(history_all, 'Epoch', 'Accuracy', ('main_acc', 'aux1_acc', 'aux2_acc'))
# show_train_history(history_all, 'Epoch', 'Loss', ('main_loss', 'aux1_loss', 'aux2_loss'))

