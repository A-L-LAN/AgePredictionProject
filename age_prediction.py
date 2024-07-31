import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define paths and constants
DATA_DIR = 'data/UTKFace/'  # Change this to your dataset path
IMG_SIZE = 128  # Resize to 128x128
BATCH_SIZE = 32

# Function to extract age from filename
def get_age(filename):
    return int(filename.split('_')[0])

# Custom data generator function
def load_data(data_dir, img_size):
    images = []
    ages = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            img_path = os.path.join(data_dir, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            age = get_age(file)
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

# Load the dataset
images, ages = load_data(DATA_DIR, IMG_SIZE)

# Normalize images
images = images / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1)  # Output layer with 1 neuron for regression (age prediction)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Mean Absolute Error: {mae}")

# Plot training history
plt.plot(history.history['mae'], label='MAE (training)')
plt.plot(history.history['val_mae'], label='MAE (validation)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
