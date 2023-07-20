import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import os
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Setting seed for reproducibility
np.random.seed(42) 

# Defining paths to folders with train, test, and validation data
train_data = 'train'
test_data = 'test'
validation_data = 'validation'
classes = os.listdir(train_data)

# Setting image width and height
img_width, img_height = 224, 224

# Creating ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    classes=classes,
    shuffle=True
)

# Creating ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(
    validation_data,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    classes=classes,
    shuffle=True
)

# Creating ImageDataGenerator for test data (no data augmentation)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    classes=classes,
    shuffle=False 
)

# Creating a Sequential model
model = Sequential()

# Using a pretrained ResNet50 model with weights from 'imagenet'
pretrained_model = tf.keras.applications.resnet50.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(224, 224, 3),
                    input_tensor=None,
                    pooling=None,
                    classes=7)

# Freezing the layers of the pretrained model so they won't be trained
for layer in pretrained_model.layers:
    layer.trainable = False

# Adding the pretrained ResNet50 model to our sequential model
model.add(pretrained_model)

# Flattening the output of the pretrained model
model.add(Flatten())

# Adding a dense layer with 1024 units and ReLU activation
model.add(Dense(1024, activation='relu'))

# Adding the output layer with 7 units (one for each fruit class) and softmax activation
model.add(Dense(7, activation='softmax'))

# Compiling the model with the Adam optimizer and categorical_crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Setting up callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
model_checkpoint = ModelCheckpoint('weights2.hdf5', monitor='val_accuracy', save_best_only=True)

# Training the model using the ImageDataGenerators and saving the training history
history = model.fit(train_generator, validation_data=validation_generator, epochs=1, batch_size=32, callbacks=[early_stopping, model_checkpoint])

# Plotting the training and validation accuracy over epochs
fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# Load the best weights from the checkpoint (highest validation accuracy)
model.load_weights('weights2.hdf5')

# Evaluating the model on the test data
loss, accuracy = model.evaluate(test_generator)

# Making predictions on the test data
predictions = model.predict(test_generator)

# Choosing the class label with the highest probability as the predicted class
predicted_labels = np.argmax(predictions, axis=1)

# Extracting true labels from the test generator
true_labels = test_generator.classes

# Converting numerical labels to fruit names
predicted_labels_text = [classes[label] for label in predicted_labels]
true_labels_text = [classes[label] for label in true_labels]
    
# Creating a DataFrame to compare predicted labels with true labels for a subset of data
comparison_df2 = pd.DataFrame({"Predictions": predicted_labels_text, "True Values": true_labels_text})
random_rows = comparison_df2.sample(n=30)
print(random_rows)
