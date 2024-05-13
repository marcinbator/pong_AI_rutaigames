import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ścieżka do folderu z obrazami
images_dir = 'images'

# Utwórz katalogi dla danych treningowych i testowych, jeśli nie istnieją
train_dir = 'train_images'
test_dir = 'test_images'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Wczytaj listę plików obrazów
image_files = os.listdir(images_dir)

# Przetasuj listę plików obrazów
random.shuffle(image_files)

# Podział danych na zestawy treningowy i testowy (np. 80% treningowy, 20% testowy)
split_ratio = 0.8
train_size = int(len(image_files) * split_ratio)

# Przenieś obrazy do odpowiednich katalogów
for i, image_file in enumerate(image_files):
    src_path = os.path.join(images_dir, image_file)
    if i < train_size:
        dst_path = os.path.join(train_dir, image_file)
    else:
        dst_path = os.path.join(test_dir, image_file)
    shutil.copy(src_path, dst_path)

# Przygotowanie danych treningowych i testowych za pomocą ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(144, 256),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(144, 256),
    batch_size=32,
    class_mode='binary')

# Budowa modelu CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(144, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // 32)

# Wykres dokładności i straty
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()