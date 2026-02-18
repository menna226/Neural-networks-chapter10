# Building an Image Recognition Neural Network (MNIST)

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras 

# Setting random seeds for reproducibility
np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(1)

# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Size of the training set:", len(x_train))
print("Size of the testing set:", len(x_test))

# Visualize a single sample
plt.imshow(x_train[5], cmap='Greys')
print("The label is", y_train[5])

# Visualize the first 5 images with their labels
fig = plt.figure(figsize=(20,20))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='Greys')
    ax.set_title('Label: ' + str(y_train[i]))
plt.show() 

# -----------------------------
# Pre-processing the data
# Flatten images from 28x28 to 784 features
x_train_reshaped = x_train.reshape(-1, 28*28)
x_test_reshaped = x_test.reshape(-1, 28*28)

# Convert labels to one-hot encoded vectors
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -----------------------------
# Building the Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28*28,)))  # First hidden layer
model.add(Dropout(0.2))                                         # Dropout for regularization
model.add(Dense(64, activation='relu'))                         # Second hidden layer
model.add(Dropout(0.2))                                         # Dropout for regularization
model.add(Dense(10, activation='softmax'))                      # Output layer (10 classes)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train_reshaped, y_train_cat, epochs=10, batch_size=10)

# -----------------------------
# Making Predictions
predictions_vector = model.predict(x_test_reshaped)
predictions = [np.argmax(pred) for pred in predictions_vector]

# Visualize predictions for specific test images
plt.imshow(x_test[4], cmap='Greys')
plt.xticks([]); plt.yticks([])
print("The label is", y_test[4])
print("The prediction is", predictions[4])

plt.imshow(x_test[18], cmap='Greys')
plt.xticks([]); plt.yticks([])
print("The label is", y_test[18])
print("The prediction is", predictions[18])

# -----------------------------
# Evaluating Accuracy
num_correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        num_correct += 1

print("The model is correct", num_correct, "times out of", len(y_test))
print("The accuracy is", num_correct / len(y_test))