import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np


BATCH_SIZE = 16
IMG_HEIGHT = 128 
IMG_WIDTH = 128


IMAGES_DIR = 'enter your image file location'
MASKS_DIR = 'enter your mask file location'




image_filenames = sorted(os.listdir(IMAGES_DIR))
mask_filenames = sorted(os.listdir(MASKS_DIR))


if len(image_filenames) != len(mask_filenames):
    raise ValueError(f"Mismatch! Found {len(image_filenames)} images and {len(mask_filenames)} masks. They must be equal.")

image_paths = [os.path.join(IMAGES_DIR, f) for f in image_filenames]
mask_paths = [os.path.join(MASKS_DIR, f) for f in mask_filenames]

print(f"Successfully found {len(image_paths)} image-mask pairs!")


def process_path(image_path, mask_path):
    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0 

    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1) 
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    mask = mask / 255.0 
    
    return img, mask


dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)


DATASET_SIZE = len(image_paths)
train_size = int(0.8 * DATASET_SIZE)

dataset = dataset.shuffle(buffer_size=500, seed=42)
train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    
    u4 = layers.UpSampling2D((2, 2))(c3)
    concat4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    concat5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat5)

    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return models.Model(inputs, outputs)

model = build_unet((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


EPOCHS = 20

print("\nStarting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)


model.save('myModel.keras')
print("\nModel saved as 'myModel.keras'")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Train Acc')
plt.plot(range(EPOCHS), val_acc, label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Train Loss')
plt.plot(range(EPOCHS), val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()