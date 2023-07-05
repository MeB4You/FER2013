import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import sys
from random_eraser import get_random_eraser

seed = 1234

FERPLUS_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

def preprocess_data():
    # load FER2013 with FER2013Plus labels
    data = pd.read_csv('fer2013.csv')
    labels = pd.read_csv('fer2013new.csv')

    # "Training" = Training data, "PublicTest" = validation data, "PrivateTest" = testing data
    # Traing = 28709, Validation = Testing = 3589
    # Training
    train_data   = data.loc[labels['Usage'] == "Training"].reset_index(drop = True)
    train_labels = labels.loc[labels['Usage'] == "Training"].reset_index(drop = True)
    X_train,y_train = convert_df_to_array(train_data,train_labels)
    X_train,y_train = clean_data_and_normalization(X_train,y_train)
    # Validation 
    val_data   = data.loc[labels['Usage'] == "PublicTest"].reset_index(drop = True)
    val_labels = labels.loc[labels['Usage'] == "PublicTest"].reset_index(drop = True)
    X_val,y_val = convert_df_to_array(val_data,val_labels)
    X_val,y_val = clean_data_and_normalization(X_val,y_val)
    # Testing
    test_data   = data.loc[labels['Usage'] == "PrivateTest"].reset_index(drop = True)
    test_labels = labels.loc[labels['Usage'] == "PrivateTest"].reset_index(drop = True)
    X_test,y_test = convert_df_to_array(test_data,test_labels)
    X_test,y_test = clean_data_and_normalization(X_test,y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def convert_df_to_array(Xdf,ydf):
    w = 48
    h = 48
    n_samples = len(Xdf)
    y = np.array(ydf[FERPLUS_labels])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(Xdf['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))
        
    return X, y
def clean_data_and_normalization(X, y):
    X,y = drop_unknown_and_NF(X,y)
    label = []
    for i in range(0,len(y)):
        max_label_index = np.argmax(y[i]) 
        label.append(max_label_index)
    y = np.array(label)
    # Normalization
    X = X / 255.0

    return X, y
def drop_unknown_and_NF(X,y):
    indices_to_remove = []
    for i in range(0,len(y)):
        max_label = np.argmax(y[i],axis = 0)
        if(max_label == 9 or max_label == 8):
            indices_to_remove.append(i)
    X = np.delete(X,indices_to_remove,axis = 0)
    y = np.delete(y,indices_to_remove,axis = 0)
    y = y[:, :-2]
    return X,y

def data_augmentation(x_train, mode = "Random Erasing"):
    shift = 0.3
    if mode == "Random Erasing":
        datagen = ImageDataGenerator(
            rotation_range=10,
            height_shift_range=shift,
            width_shift_range=shift,
            horizontal_flip=True,
            preprocessing_function= get_random_eraser(v_l=0, v_h=1,pixel_level= False),
            fill_mode="constant")
    else: 
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            height_shift_range=shift,
            width_shift_range=shift,
            fill_mode="nearest")
        
    datagen.fit(x_train)
    return datagen

def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
    plt.show()

def show_images(x_train,y_train):
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap='gray')
        plt.xlabel(FERPLUS_labels[y_train[i]])
    plt.show()

def get_dataset():
    return preprocess_data()