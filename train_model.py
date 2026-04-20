import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 16
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 100

IMAGES_DIR = 'your_images_directory_here'
MASKS_DIR = 'your_masks_directory_here'

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
    weight = y_true * alpha * K.pow(1.0 - y_pred, gamma) + \
             (1.0 - y_true) * (1.0 - alpha) * K.pow(y_pred, gamma)
    return K.mean(weight * cross_entropy)

def robust_focal_dice_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

image_filenames = sorted(os.listdir(IMAGES_DIR))
mask_filenames = sorted(os.listdir(MASKS_DIR))

image_paths = [os.path.join(IMAGES_DIR, f) for f in image_filenames]
mask_paths = [os.path.join(MASKS_DIR, f) for f in mask_filenames]

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

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask

train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))
val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def residual_block(inputs, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    shortcut = inputs
    if K.int_shape(inputs)[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_deep_resunet(input_shape):
    inputs = layers.Input(shape=input_shape)

    r1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(r1)

    r2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(r2)

    r3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(r3)

    r4 = residual_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(r4)
    p4 = layers.Dropout(0.3)(p4)

    r5 = residual_block(p4, 512)
    p5 = layers.MaxPooling2D((2, 2))(r5)
    p5 = layers.Dropout(0.4)(p5)

    b = residual_block(p5, 1024)

    u5 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(b)
    c5 = layers.concatenate([u5, r5])
    d5 = residual_block(c5, 512)

    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d5)
    c4 = layers.concatenate([u4, r4])
    d4 = residual_block(c4, 256)

    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d4)
    c3 = layers.concatenate([u3, r3])
    d3 = residual_block(c3, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d3)
    c2 = layers.concatenate([u2, r2])
    d2 = residual_block(c2, 64)

    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    c1 = layers.concatenate([u1, r1])
    d1 = residual_block(c1, 32)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)

    return models.Model(inputs, outputs)

model = build_deep_resunet((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
              loss=robust_focal_dice_loss,
              metrics=[dice_coef])

checkpoint = callbacks.ModelCheckpoint(
    'my_custom_resunet.keras',
    monitor='val_dice_coef',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_run = len(dice)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_run + 1), dice, label='Train Dice')
plt.plot(range(1, epochs_run + 1), val_dice, label='Val Dice')
plt.legend()
plt.title('Dice Coefficient')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_run + 1), loss, label='Train Loss')
plt.plot(range(1, epochs_run + 1), val_loss, label='Val Loss')
plt.legend()
plt.title('Robust Focal-Dice Loss')

plt.savefig('custom_resunet_training.png')