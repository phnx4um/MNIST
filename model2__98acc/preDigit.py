import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import os

def loadModel():
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("mnist_model.h5")
    return model



model = loadModel()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_test = to_categorical(y_test, num_classes=10)
X_test = X_test.reshape(X_test.shape[0], 28*28)

X_test = X_test.astype('float32')
X_test /= 255

loss_and_metrics = model.evaluate(X_test, y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])


img = X_test[1001]
img = img.reshape(1,28*28)
img_class = model.predict_classes(img)
prediction = img_class[0]
classname = img_class[0]
print("Class: ",classname)


img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()





