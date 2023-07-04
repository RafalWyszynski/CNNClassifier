import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def show_image(image, title='Title', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.show()
    
training_files = 'train'
validation_files = 'validation'
classes = ['apple']

images = []
labels = []

for fruit in classes:
    
    #Creating paths for each fruit
    training_fruit_path = os.path.join(training_files, fruit)
    training_images_amount = os.listdir(training_fruit_path)
    
    for fruit_image in training_images_amount:
        photo_path = os.path.join(training_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        #image = image.astype('uint8')
        image = image[:,:,:3]
        image = cv2.resize(image, (800, 800))
        images.append(image)
        labels.append(fruit)
    
show_image(images[21])        
imagehist = images[19]
    
gray = color.rgb2gray(imagehist)
from skimage import exposure
image_adapteq = exposure.equalize_adapthist(imagehist, clip_limit=0.01)
show_image(imagehist, 'Original')
show_image(image_adapteq, 'Adaptive Equalized')

images = np.array(images)     
labels = np.array(labels)