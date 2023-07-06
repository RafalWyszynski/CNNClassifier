import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rotate
from skimage.filters import sobel
from skimage import exposure
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(50)

#Adding easier function to show images
def show_image(image, title='Title', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.show()

############################################## Images Loading and Rescaling #################################################
training_files = 'train'
testing_files = 'test'
validation_files = 'validation'
classes = ['apple', 'banana', 'kiwi', 'orange', 'watermelon'] #, 'mango', 'pear']

images = []
labels = []
val_images = []
val_labels = []

for fruit in classes:
    
    #Creating paths for each fruit
    training_fruit_path = os.path.join(training_files, fruit)
    testing_fruit_path = os.path.join(testing_files, fruit)
    validation_fruit_path = os.path.join(validation_files, fruit)
    
    #Checking the number of images in each folder
    training_images_amount = os.listdir(training_fruit_path)
    testing_images_amount = os.listdir(testing_fruit_path)
    validation_images_amount = os.listdir(validation_fruit_path)
    
    #Adding images and labels to lists
    for fruit_image in training_images_amount:
        photo_path = os.path.join(training_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (400, 400))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in testing_images_amount:
        photo_path = os.path.join(testing_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (400, 400))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in validation_images_amount:
        photo_path = os.path.join(validation_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (400, 400))
        val_images.append(image)
        val_labels.append(fruit)                        

######################################################## Images Preprocessing #################################################

#Adding 200 randomly chosen horizontally flipped versions of images
for i in range (200):
    random_index = random.randint(0, len(images)-1)
    horizontally_flipped = np.fliplr(images[random_index])
    images.append(horizontally_flipped)
    labels.append(labels[random_index])

#Adding 200 randomly choosen vertically flipped versions of images
for i in range (200):
    random_index = random.randint(0, len(images)-1)
    vertically_flipped = np.flipud(images[random_index])
    images.append(vertically_flipped)
    labels.append(labels[random_index])

#Adding 200 randomly chosen rotated versions of images
for i in range (100):
    random_index = random.randint(0, len(images)-1)
    rotated_image = rotate(images[random_index], -90)
    images.append(rotated_image)
    labels.append(labels[random_index])
    
#Adding 200 randomly chosen images with equalized histogram (using CLAHE method)
for i in range(100):
    random_index = random.randint(0, len(images) - 1)
    histogram_equalized = exposure.equalize_adapthist(images[random_index])
    images.append(histogram_equalized)
    labels.append(labels[random_index])
 
#Adding 200 randomly chosen images with detected contours (using Sobel filter)    
for i in range(100):
    random_index = random.randint(0, len(images) - 1)
    edge_sobel = sobel(images[random_index])
    images.append(edge_sobel)
    labels.append(labels[random_index])
    
#Adding 200 photos with 100x100 green rectangles
for i in range(100):
    random_index = random.randint(0, len(images) - 1)
    random_x = random.randint(0, 700)
    random_y = random.randint(0,700)
    image_with_rectangle = images[random_index]
    image_with_rectangle[random_x:random_x+100, random_y:random_y+100, :] = [0, 255, 0]
    images.append(image_with_rectangle)
    labels.append(labels[random_index])

#Converting images and their labels to NumPy arrays
images = np.array(images)     
labels = np.array(labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

######################################################## Model #################################################

#Converting fruits names using One-Hot Encoding
labels = pd.Categorical(labels).codes
labels = to_categorical(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

#Defining model
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=3, activation=activation, input_shape=(400,400,3), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(16, kernel_size=3, activation=activation, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16, kernel_size=3, activation=activation, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, kernel_size=3, activation=activation, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(8, kernel_size=3, activation=activation, padding='same')) 

    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Initalizing model
model = KerasClassifier(build_fn=create_model, epochs=1000, batch_size=32)

#Creating parameters dictionary to Randomzied Search
parameters = dict(optimizer=['sgd', 'adam'], activation=['relu', 'leaky_relu', 'tanh'], batch_size=[16,32])

#Initializing Randomized Search
random_search = RandomizedSearchCV(model, parameters, cv=3, n_iter=10)

#Initializing Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
model_checkpoint = ModelCheckpoint('weights2.hdf5', monitor='val_accuracy', save_best_only=True)

# Searching for the best parameters and showing results
random_search_results = random_search.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
print('Best: {} using: {}'.format(random_search_results.best_score_, random_search_results.best_params_))