import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset123
train_data_path = '/path/to/train_data'
test_data_path = '/path/to/test_data'

# Define the dimensions of the input images and the number of classes
input_shape = (64, 64, 3)
num_classes = 2

# Data augmentation for training set
train_data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Data augmentation for test set (only rescaling)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_data = train_data_generator.flow_from_directory(train_data_path, target_size=input_shape[:2], batch_size=32, class_mode='categorical')

# Load and preprocess test data
test_data = test_data_generator.flow_from_directory(test_data_path, target_size=input_shape[:2], batch_size=32, class_mode='categorical')

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=test_data)

# Save the trained model
model.save('pattern_recognition_model.h5')
