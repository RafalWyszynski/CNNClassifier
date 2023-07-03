import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate
from skimage.filters import sobel
from skimage import exposure

#Adding easier function to show images
def show_image(image, title='Title', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.show()

############################################## Images Loading and Rescaling #################################################
training_files = 'train'
testing_files = 'test'
validation_files = 'validation'
classes = ['apple', 'banana', 'kiwi', 'mango', 'orange', 'pear', 'watermelon']

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
        image = cv2.resize(image, (800, 800))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in testing_images_amount:
        photo_path = os.path.join(testing_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (800, 800))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in validation_images_amount:
        photo_path = os.path.join(validation_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (800, 800))
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
for i in range (200):
    random_index = random.randint(0, len(images)-1)
    rotated_image = rotate(images[random_index], -90)
    images.append(rotated_image)
    labels.append(labels[random_index])
    
#Adding 200 randomly chosen images with equalized histogram (using CLAHE method)
for i in range(200):
    random_index = random.randint(0, len(images) - 1)
    histogram_equalized = exposure.equalize_adapthist(images[random_index])
    images.append(histogram_equalized)
    labels.append(labels[random_index])
 
#Adding 200 randomly chosen images with detected contours (using Sobel filter)    
for i in range(200):
    random_index = random.randint(0, len(images) - 1)
    edge_sobel = sobel(images[random_index])
    images.append(edge_sobel)
    labels.append(labels[random_index])
    
#Converting images and their labels to NumPy arrays
images = np.array(images)     
labels = np.array(labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)


######################################################## Model #################################################

