import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np

BATCH_SIZE = 16
IMG_HEIGHT = 128 
IMG_WIDTH = 128
EPOCHS = 40

IMAGES_DIR = 'your_images_directory_here'
MASKS_DIR = 'your_masks_directory_here'

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

image_filenames = sorted(os.listdir(IMAGES_DIR))
mask_filenames = sorted(os.listdir(MASKS_DIR))

if len(image_filenames) != len(mask_filenames):
    raise ValueError(f"Mismatch! Found {len(image_filenames)} images and {len(mask_filenames)} masks.")

image_paths = [os.path.join(IMAGES_DIR, f) for f in image_filenames]
mask_paths = [os.path.join(MASKS_DIR, f) for f in mask_filenames]

print(f"Successfully found {len(image_paths)} image-mask pairs!")

DATASET_SIZE = len(image_paths)
train_size = int(0.8 * DATASET_SIZE)

train_img_paths, val_img_paths = image_paths[:train_size], image_paths[train_size:]
train_mask_paths, val_mask_paths = mask_paths[:train_size], mask_paths[train_size:]

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0 

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1) 
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = mask / 255.0 
    
    return img, mask

train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def conv_block(inputs, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.3)(p4)

    c5 = conv_block(p4, 512)

    u6 = layers.UpSampling2D((2, 2))(c5)
    concat6 = layers.concatenate([u6, c4])
    c6 = conv_block(concat6, 256)

    u7 = layers.UpSampling2D((2, 2))(c6)
    concat7 = layers.concatenate([u7, c3])
    c7 = conv_block(concat7, 128)

    u8 = layers.UpSampling2D((2, 2))(c7)
    concat8 = layers.concatenate([u8, c2])
    c8 = conv_block(concat8, 64)

    u9 = layers.UpSampling2D((2, 2))(c8)
    concat9 = layers.concatenate([u9, c1])
    c9 = conv_block(concat9, 32)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return models.Model(inputs, outputs)

model = build_unet((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss=bce_dice_loss, 
              metrics=[dice_coef])

print("\nStarting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

model.save('myModel.keras')
print("\nModel saved as 'myModel.keras'")

dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), dice, label='Train Dice Score')
plt.plot(range(EPOCHS), val_dice, label='Val Dice Score')
plt.legend()
plt.title('Dice Coefficient')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Train Loss')
plt.plot(range(EPOCHS), val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()