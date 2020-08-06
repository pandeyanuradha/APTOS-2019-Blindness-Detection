import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.applications.densenet import DenseNet121,DenseNet169
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model
dictionary={0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.shape

df_test.shape

df_train.isnull().sum()

df_test.isnull().sum()

data_copy=df_train.copy()

data_copy['diagnosis']=data_copy['diagnosis'].map(dictionary)

data_copy['diagnosis'].value_counts()

chat_data = data_copy.diagnosis.value_counts()
chat_data.plot(kind='bar');
plt.title('Sample Per Class');
plt.show()
plt.pie(chat_data, autopct='%1.1f%%', shadow=True,labels=dictionary.items())
plt.title('Per class sample Percentage');
plt.show()

def view_fundus_images(images, title = ''):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))   
    for im in range(0, height * width):
        # open image
        image = Image.open(os.path.join(r"/home/ubuntu/train_images",images[im] + '.png'))
        i = im // width
        j = im % width
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')

    # set suptitle
    plt.suptitle(title)
    plt.show()
view_fundus_images(df_train[df_train['diagnosis'] == 0][:10].id_code.values, title = 'Images without DR')

view_fundus_images(df_train[df_train['diagnosis'] == 1][:10].id_code.values, title = 'Images with Mild condition')

view_fundus_images(df_train[df_train['diagnosis'] == 2][:10].id_code.values, title = 'Images with Moderate condition')

view_fundus_images(df_train[df_train['diagnosis'] == 2][:10].id_code.values, title = 'Images with Severe condition')

view_fundus_images(df_train[df_train['diagnosis'] == 4][:10].id_code.values, title = 'Images with Proliferative DR')

def img_size(df,train=True):
    if train:
        path=r'/home/ubuntu/train_images/'
    else:
        path=r'/home/ubuntu/test_images/'
        
    widths=[]
    heights=[]
    
    images=df.id_code
    max_im = Image.open(os.path.join(path, images[0] + '.png'))
    min_im = Image.open(os.path.join(path, images[0] + '.png'))
        
    for im in range(0, len(images)):
        image = Image.open(os.path.join(path, images[im] + '.png'))
        width, height = image.size
        
        if len(widths) > 0:
            if width > max(widths):
                max_im = image

            if width < min(widths):
                min_im = image

        widths.append(width)
        heights.append(height)
        
    return widths, heights, max_im, min_im  

train_widths, train_heights, max_train, min_train = img_size(df_train, train = True)
test_widths, test_heights, max_test, min_test = img_size(df_test, train = False)

print('Maximum width for training set is {}'.format(max(train_widths)))
print('Minimum width for training set is {}'.format(min(train_widths)))
print('Maximum height for training set is {}'.format(max(train_heights)))
print('Minimum height for training set is {}'.format(min(train_heights)))

print('Maximum width for test set is {}'.format(max(test_widths)))
print('Minimum width for test set is {}'.format(min(test_widths)))
print('Maximum height for test set is {}'.format(max(test_heights)))
print('Minimum height for test set is {}'.format(min(test_heights)))

plt.imshow(max_train)

plt.imshow(min_train)

plt.imshow(max_test)

plt.imshow(min_test)

df_train['diagnosis']=df_train['diagnosis'].map(dictionary)

df_train.head()

df_train['diagnosis'].value_counts()

df_train['diagnosis'] = df_train['diagnosis'].astype('str')
df_train['id_code'] = df_train['id_code'].astype(str)+'.png'

datagen=ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batch_size = 16
image_size = 96

train_gen=datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=r'/home/ubuntu/train_images/',
    x_col="id_code",
    y_col="diagnosis",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    save_format='jpeg',
    target_size=(image_size,image_size),
    subset='training')

val_gen=datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=r'/home/ubuntu/train_images/',
    x_col="id_code",
    y_col="diagnosis",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",
    save_format='jpeg',
    target_size=(image_size,image_size),
    subset='validation')

df_train.head()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, GaussianNoise, GaussianDropout
from keras.layers import Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.constraints import maxnorm
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers, optimizers


# In[35]:


def build_model():
    # create model
    model = Sequential()
    model.add(GaussianDropout(0.3,input_shape=[96,96,3]))
    model.add(Conv2D(15, (3, 3), input_shape=[96,96,3], activation='relu'))
    model.add(GaussianDropout(0.3))
    model.add(Conv2D(30, (5, 5), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (5, 5), activation='relu'))
    model.add(Conv2D(50, (7, 7), activation='relu'))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.01)))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])
    return model

model = build_model()

from keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 20)
mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)

model.summary()

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50,
                    callbacks = [es, mc], 
                    verbose=1)

model.evaluate_generator(generator=val_gen,steps=STEP_SIZE_VALID)
