import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in testing_images_amount:
        photo_path = os.path.join(testing_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in validation_images_amount:
        photo_path = os.path.join(validation_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        val_images.append(image)
        val_labels.append(fruit)    
                       
#Converting images and their labels to NumPy arrays
images = np.array(images)     
labels = np.array(labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

        
print(len(images))
print(len(labels))
print(val_labels[50])    
plt.imshow(val_images[50])
plt.show()

        
    