import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

custom_objects_dict = {
    'robust_focal_dice_loss': robust_focal_dice_loss,
    'dice_coef': dice_coef
}

model = tf.keras.models.load_model('my_custom_resunet.keras', custom_objects=custom_objects_dict)

def detect_and_draw_boxes(image_path, model, img_height=128, img_width=128):
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    display_img = original_img.copy()
    
    input_img = cv2.resize(original_img, (img_width, img_height))
    input_img = input_img / 255.0
    input_tensor = np.expand_dims(input_img, axis=0)
    
    prediction = model.predict(input_tensor)[0]
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    binary_mask_resized = cv2.resize(binary_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(display_img, 'Plastic', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
    plt.figure(figsize=(10, 10))
    plt.imshow(display_img)
    plt.axis('off')
    plt.savefig('detected_boxes.png', bbox_inches='tight')

TEST_IMAGE_PATH = 'path_to_your_test_image.jpg'
detect_and_draw_boxes(TEST_IMAGE_PATH, model)