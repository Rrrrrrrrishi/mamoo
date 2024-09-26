import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Function to load data from a specified directory
def load_data(data_dir):
    images = []
    labels = []
    
    for label in ['benign', 'malignant']:
        folder_path = os.path.join(data_dir, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (100, 100))  # Resize for consistency
                images.append(img)
                labels.append(0 if label == 'benign' else 1)  # Binary labels

    return np.array(images), np.array(labels)

# Load training data
train_data_dir = r"C:\Users\hp\OneDrive\Desktop\code\dataset\train"  # Update this to your train dataset path
X_train, y_train = load_data(train_data_dir)

# Load testing data
test_data_dir = r"C:\Users\hp\OneDrive\Desktop\code\dataset\train"  # Update this to your test dataset path
X_test, y_test = load_data(test_data_dir)

# Normalize the images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('my_model.keras')

