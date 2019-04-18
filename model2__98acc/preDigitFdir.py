import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps 
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.preprocessing import image
import os

def loadModel():
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("mnist_model.h5")
    return model

model = loadModel()

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

path = 'images'
list = os.listdir(path)
print(list)

marks = []

for i in range(5):
    img_path = "images/" + str(i) +".jpeg"
    print(img_path)
    img = image.load_img(path=img_path,color_mode = "grayscale",target_size=(28,28,1))
    img = image.img_to_array(img)
    img = img.reshape((1,784))
    img = 1 - img

    img_class = model.predict_classes(img)
    prediction = img_class[0]
    classname = img_class[0]
    print(classname) 
    marks.append(classname)

print(marks)
img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()





