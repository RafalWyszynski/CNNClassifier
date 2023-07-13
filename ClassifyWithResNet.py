import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

training_files = 'train'
testing_files = 'test'
validation_files = 'validation'
classes = ['apple', 'banana', 'kiwi', 'orange', 'watermelon', 'mango', 'pear']

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
        image = cv2.resize(image, (224,224))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in testing_images_amount:
        photo_path = os.path.join(testing_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (224,224))
        images.append(image)
        labels.append(fruit)
        
    for fruit_image in validation_images_amount:
        photo_path = os.path.join(validation_fruit_path, fruit_image)
        image = plt.imread(photo_path)
        image = image[:,:,:3]
        image = cv2.resize(image, (224,224))
        val_images.append(image)
        val_labels.append(fruit)                        

#Converting images and their labels to NumPy arrays
images = np.array(images)     
labels = np.array(labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

#Converting fruits names using One-Hot Encoding
labels = pd.Categorical(labels).codes
labels = to_categorical(labels)
val_labels = pd.Categorical(val_labels).codes
val_labels = to_categorical(val_labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

model = Sequential()

pretrained_model = tf.keras.applications.resnet50.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224,224,3),
                    input_tensor=None,
                    pooling=None,
                    classes=7)

for layer in pretrained_model.layers:
        layer.trainable=False

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
model_checkpoint = ModelCheckpoint('weights2.hdf5', monitor='val_accuracy', save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32, callbacks=[early_stopping, model_checkpoint])

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()


# Load the best weights from the checkpoint
model.load_weights('weights2.hdf5')

# Evaluating model
loss, accuracy = model.evaluate(val_images, val_labels)

# Predicting values for Validation Images
predictions = model.predict(val_images)

# Choosing prediction with highest value
predicted_labels = np.argmax(predictions, axis=1)
    
# Converting One-Hot Encoded labels back to normal array
true_labels = np.argmax(val_labels, axis=1)

# Converting numerical labels to fruits names
predicted_labels_text = []
for label in predicted_labels:
    if label == 0:
        predicted_labels_text.append('apple')
    elif label == 1:
        predicted_labels_text.append('banana')
    elif label == 2:
        predicted_labels_text.append('kiwi')
    elif label == 3:
        predicted_labels_text.append('orange')
    elif label == 4:
        predicted_labels_text.append('watermelon')
    elif label == 5:
        predicted_labels_text.append('mango')
    elif label == 6:
        predicted_labels_text.append('pear')

true_labels_text = []
for label in true_labels:
    if label == 0:
        true_labels_text.append('apple')
    elif label == 1:
        true_labels_text.append('banana')
    elif label == 2:
        true_labels_text.append('kiwi')
    elif label == 3:
        true_labels_text.append('orange')
    elif label == 4:
        true_labels_text.append('watermelon')
    elif label == 5:
        true_labels_text.append('mango')
    elif label == 6:
        true_labels_text.append('pear')
    

# Creating DataFrame to Compare few Predicted Labels with True Labels
comparison_df2 = pd.DataFrame({"Predictions": predicted_labels_text, "True Values": true_labels_text})
print(comparison_df2.iloc[15:30, :])