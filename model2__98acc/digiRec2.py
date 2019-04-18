import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import os
def getData():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255
    
    #plt.imshow(X_train[0][:,:,0])
    #plt.show()
    return X_train, y_train, X_test, y_test

def trainModel(X_train, y_train, X_test, y_test):
    # training parameters
    batch_size = 32
    epochs = 20
    # create model and add layers
    model = Sequential()

    #_______PREVIOUS MODEL________
    #model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    #model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    #model.add(MaxPool2D(pool_size=(2, 2)))
    #model.add(Dropout(rate=0.25))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPool2D(pool_size=(2, 2)))
    #model.add(Dropout(rate=0.25))
    #model.add(Flatten())
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(rate=0.5))
    #model.add(Dense(10, activation='softmax'))
    # define model optimizer and callback function
	
    #_______NEW MODEL________
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))


    ## compile model define loss, optimizer and metrics
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
   


    #____NEW COMPILATION_____
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    
    # training the model and saving metrics in history
    history = model.fit(X_train, y_train,
         batch_size=batch_size, epochs=epochs,
          verbose=2,
         validation_data=(X_test, y_test))

    loss_and_metrics = model.evaluate(X_test, y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])
    
    # Save model structure and weights
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('mnist_model.h5')
    return model

def loadModel():
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("mnist_model.h5")
    return model

X_train, y_train, X_test, y_test = getData()

if(not os.path.exists('mnist_model.h5')):
    model = trainModel(X_train, y_train, X_test, y_test)
    print('trained model')
    print(model.summary())
else:
    model = loadModel()
    print('loaded model')
    print(model.summary())
