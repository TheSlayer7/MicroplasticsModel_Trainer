import os
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tifffile
from keras import callbacks, layers, models

# Reduce OpenCV TIFF metadata warning noise.
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
if hasattr(cv2, 'setLogLevel'):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Model/input settings.
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4

# Patch sampler settings for tiny particles.
PATCH_H = 384
PATCH_W = 384
POSITIVE_PATCH_PROB = 0.40
HARD_NEG_PATCH_PROB = 0.15
MIN_POS_PIXELS = 12
HARD_NEG_ATTEMPTS = 32
MAX_POS_RATIO_IN_PATCH = 0.05

# Staged domain curriculum.
# Epoch ranges are [start, end).
CURRICULUM = [
    {'start': 0, 'end': 10, 'clam_prob': 0.40, 'pos_weight_cap': 150.0},
    {'start': 10, 'end': 30, 'clam_prob': 0.40, 'pos_weight_cap': 150.0},
    {'start': 30, 'end': 10_000, 'clam_prob': 0.40, 'pos_weight_cap': 150.0},
]

# Data folders.
DATA_DIR = Path('data')
SPIKED_IMAGES_DIR = DATA_DIR / 'spiked_fl'
SPIKED_MASKS_DIR = DATA_DIR / 'spiked_mask'
CLAM_IMAGES_DIR = DATA_DIR / 'clam_fl'
CLAM_MASKS_DIR = DATA_DIR / 'clam_mask'
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}

# Outputs.
MODEL_OUTPUT = 'my_custom_resunet.keras'
TRAINING_PLOT = 'custom_resunet_training.png'
CALIBRATION_OUTPUT = 'detection_calibration.json'

# Dynamic positive class weight used by loss.
POSITIVE_PIXEL_WEIGHT_VAR = tf.Variable(220.0, trainable=False, dtype=tf.float32, name='pos_weight')


def _read_image(path):
    p = Path(path)
    if p.suffix.lower() in {'.tif', '.tiff'}:
        arr = tifffile.imread(str(p))
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'Failed to read image: {path}')

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
        if p.suffix.lower() not in {'.tif', '.tiff'}:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return arr


def _read_mask(path):
    p = Path(path)
    if p.suffix.lower() in {'.tif', '.tiff'}:
        arr = tifffile.imread(str(p))
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'Failed to read mask: {path}')
    if arr.ndim == 3:
        arr = arr[..., 0]

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)

    # Strict polarity: standardize_masks.py should already ensure this.
    return (arr > 0.5).astype(np.uint8)


def _list_files(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f'Directory not found: {directory}')
    return sorted(str(p) for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _pair_by_stem(images, masks):
    img_by = {Path(p).stem: p for p in images}
    mask_by = {Path(p).stem: p for p in masks}
    common = sorted(set(img_by).intersection(mask_by))
    if not common:
        raise ValueError('No matching image/mask pairs by filename stem.')
    return [(img_by[s], mask_by[s]) for s in common]


def _split_pairs(pairs, val_ratio=0.2):
    rng = np.random.default_rng(RANDOM_SEED)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_ratio))) if len(pairs) > 2 else 1
    n_val = min(n_val, len(pairs) - 1) if len(pairs) > 1 else 1
    return pairs[:-n_val], pairs[-n_val:]


def _warm_spot_map(img):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    warm = np.maximum(0.0, r - 0.55 * g - 0.45 * b) + 0.6 * np.maximum(0.0, g - 0.75 * b)
    return np.clip(warm, 0.0, 1.0)


def _enhance_warm_spots_tf(image):
    r = image[..., 0:1]
    g = image[..., 1:2]
    b = image[..., 2:3]
    warm = tf.nn.relu(r - 0.55 * g - 0.45 * b) + 0.6 * tf.nn.relu(g - 0.75 * b)
    warm = tf.clip_by_value(warm, 0.0, 1.0)

    r_boost = tf.clip_by_value(r + 0.30 * warm, 0.0, 1.0)
    g_boost = tf.clip_by_value(g + 0.15 * warm, 0.0, 1.0)
    b_damp = tf.clip_by_value(0.90 * b, 0.0, 1.0)
    return tf.concat([r_boost, g_boost, b_damp], axis=-1)


def _sample_crop_coords(h, w, crop_h, crop_w):
    y0 = 0 if h == crop_h else np.random.randint(0, h - crop_h + 1)
    x0 = 0 if w == crop_w else np.random.randint(0, w - crop_w + 1)
    return y0, x0


def _resize_pair(img, mask, h=IMG_HEIGHT, w=IMG_WIDTH):
    img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r


def _sample_positive_patch(img, mask):
    h, w = mask.shape[:2]
    crop_h = min(h, PATCH_H)
    crop_w = min(w, PATCH_W)
    if h <= crop_h or w <= crop_w:
        return img, mask

    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        y0, x0 = _sample_crop_coords(h, w, crop_h, crop_w)
        return img[y0:y0 + crop_h, x0:x0 + crop_w], mask[y0:y0 + crop_h, x0:x0 + crop_w]

    max_pos_pixels = int(crop_h * crop_w * MAX_POS_RATIO_IN_PATCH)
    max_pos_pixels = max(max_pos_pixels, MIN_POS_PIXELS)

    for _ in range(HARD_NEG_ATTEMPTS):
        y, x = pos[np.random.randint(0, len(pos))]
        y0 = int(np.clip(y - crop_h // 2, 0, h - crop_h))
        x0 = int(np.clip(x - crop_w // 2, 0, w - crop_w))
        m = mask[y0:y0 + crop_h, x0:x0 + crop_w]
        m_sum = int(m.sum())
        if MIN_POS_PIXELS <= m_sum <= max_pos_pixels:
            return img[y0:y0 + crop_h, x0:x0 + crop_w], m

    y0, x0 = _sample_crop_coords(h, w, crop_h, crop_w)
    return img[y0:y0 + crop_h, x0:x0 + crop_w], mask[y0:y0 + crop_h, x0:x0 + crop_w]


def _sample_hard_negative_patch(img, mask):
    h, w = mask.shape[:2]
    crop_h = min(h, PATCH_H)
    crop_w = min(w, PATCH_W)
    if h <= crop_h or w <= crop_w:
        return img, mask

    bright = _warm_spot_map(img)
    bright_thr = float(np.quantile(bright, 0.85))

    best = None
    best_score = -1.0
    for _ in range(HARD_NEG_ATTEMPTS):
        y0, x0 = _sample_crop_coords(h, w, crop_h, crop_w)
        y1, x1 = y0 + crop_h, x0 + crop_w
        m = mask[y0:y1, x0:x1]
        if int(m.sum()) > 0:
            continue
        b = bright[y0:y1, x0:x1]
        score = float(np.mean(b > bright_thr))
        if score > best_score:
            best_score = score
            best = (y0, y1, x0, x1)

    if best is None:
        y0, x0 = _sample_crop_coords(h, w, crop_h, crop_w)
        y1, x1 = y0 + crop_h, x0 + crop_w
    else:
        y0, y1, x0, x1 = best

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _augment(img, mask):
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if np.random.random() > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    if np.random.random() > 0.5:
        delta = np.random.uniform(-0.08, 0.08)
        img = np.clip(img + delta, 0.0, 1.0)
    if np.random.random() > 0.5:
        alpha = np.random.uniform(0.85, 1.2)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * alpha + mean, 0.0, 1.0)
    if np.random.random() > 0.5:
        noise = np.random.normal(0.0, 0.02, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    return img, mask


class DomainPatchGenerator:
    def __init__(
        self,
        spiked_pairs,
        clam_pairs,
        batch_size,
        steps_per_epoch,
        training=True,
        clam_prob=0.4,
    ):
        self.spiked_pairs = spiked_pairs
        self.clam_pairs = clam_pairs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.training = training
        self.clam_prob = clam_prob

    def set_clam_prob(self, p):
        self.clam_prob = float(np.clip(p, 0.0, 1.0))

    def __iter__(self):
        while True:
            x = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
            y = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

            for i in range(self.batch_size):
                if self.training:
                    use_clam = np.random.random() < self.clam_prob
                    domain_pairs = self.clam_pairs if use_clam else self.spiked_pairs
                    img_path, mask_path = domain_pairs[np.random.randint(0, len(domain_pairs))]
                else:
                    # Validation is clam-only target domain.
                    img_path, mask_path = self.clam_pairs[np.random.randint(0, len(self.clam_pairs))]

                img = _read_image(img_path)
                mask = _read_mask(mask_path)

                if self.training:
                    r = np.random.random()
                    if int(mask.sum()) > 0 and r < POSITIVE_PATCH_PROB:
                        img, mask = _sample_positive_patch(img, mask)
                    elif r < POSITIVE_PATCH_PROB + HARD_NEG_PATCH_PROB:
                        img, mask = _sample_hard_negative_patch(img, mask)

                img, mask = _resize_pair(img, mask)

                if self.training:
                    img, mask = _augment(img, mask)

                img = _enhance_warm_spots_tf(tf.convert_to_tensor(img, dtype=tf.float32)).numpy()
                x[i] = np.clip(img, 0.0, 1.0)
                y[i, ..., 0] = (mask > 0.5).astype(np.float32)

            yield x, y


# Metrics/loss.
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def weighted_bce_loss(y_true, y_pred, pos_weight):
    eps = tf.keras.backend.epsilon()
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    w = tf.cast(pos_weight, y_pred.dtype)
    bce = -(w * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


def tversky_coef(y_true, y_pred, alpha=0.25, beta=0.75, smooth=1e-6):
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def robust_hybrid_loss(y_true, y_pred):
    pos_w = tf.minimum(POSITIVE_PIXEL_WEIGHT_VAR, tf.constant(150.0, dtype=tf.float32))
    return (
        0.45 * weighted_bce_loss(y_true, y_pred, pos_weight=pos_w)
        + 0.35 * (1.0 - tversky_coef(y_true, y_pred, alpha=0.4, beta=0.6))
        + 0.20 * dice_loss(y_true, y_pred)
    )


def residual_block(inputs, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    shortcut = inputs
    in_ch = inputs.shape[-1]
    if in_ch is None or in_ch != filters:
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

    b = residual_block(p4, 512)

    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(b)
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


def compute_fg_stats(pairs):
    ratios = []
    inverted_count = 0
    for _, mask_path in pairs:
        m = _read_mask(mask_path)
        r = float(m.mean())
        ratios.append(r)
        if r > 0.5:
            inverted_count += 1
    if not ratios:
        raise ValueError('No masks found while computing foreground stats.')
    ratios = np.asarray(ratios, dtype=np.float64)
    mean_fg = float(np.mean(ratios))
    return {
        'count': len(ratios),
        'inverted_masks': int(inverted_count),
        'mean_fg': mean_fg,
        'median_fg': float(np.median(ratios)),
        'min_fg': float(np.min(ratios)),
        'max_fg': float(np.max(ratios)),
        'tiny_lt_0_1pct': int(np.sum(ratios < 0.001)),
        'suggested_pos_weight': float(np.clip((1.0 - max(mean_fg, 1e-6)) / max(mean_fg, 1e-6), 120.0, 300.0)),
    }


class CurriculumScheduler(callbacks.Callback):
    def __init__(self, train_gen, initial_pos_weight):
        super().__init__()
        self.train_gen = train_gen
        self.initial_pos_weight = float(initial_pos_weight)

    def on_epoch_begin(self, epoch, logs=None):
        phase = CURRICULUM[-1]
        for item in CURRICULUM:
            if item['start'] <= epoch < item['end']:
                phase = item
                break

        clam_prob = float(phase['clam_prob'])
        cap = float(phase['pos_weight_cap'])
        current_w = min(self.initial_pos_weight, cap)
        self.train_gen.set_clam_prob(clam_prob)
        POSITIVE_PIXEL_WEIGHT_VAR.assign(current_w)

        tf.print(
            'Epoch', epoch + 1,
            '| clam_prob', clam_prob,
            '| pos_weight', POSITIVE_PIXEL_WEIGHT_VAR,
        )


def calibrate_thresholds(model, val_pairs):
    # Calibrate for detection inference with clam validation only.
    score_grid = np.linspace(0.08, 0.30, 23)
    color_grid = np.linspace(0.01, 0.25, 16)

    def postprocess_binary_mask(mask_u8):
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        cleaned = np.zeros_like(mask_u8)
        min_area = max(4, int(mask_u8.size * 0.00001))
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == lbl] = 1
        return cleaned

    cache = []
    for img_path, mask_path in val_pairs:
        img = _read_image(img_path)
        msk = _read_mask(mask_path).astype(np.uint8)

        inp = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        pred = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        pred = np.clip(np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        pred_r = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred_b = cv2.GaussianBlur(pred_r.astype(np.float32), (0, 0), sigmaX=1.0)
        color = cv2.GaussianBlur(_warm_spot_map(img).astype(np.float32), (0, 0), sigmaX=1.0)
        combined = np.clip(0.92 * pred_b + 0.08 * color, 0.0, 1.0)
        cache.append((combined, color, msk))

    best_dice = -1.0
    best_s = 0.42
    best_c = 0.05

    for s in score_grid:
        for c in color_grid:
            ds = []
            for comb, col, gt in cache:
                pred_m = ((comb > s) & (col > c)).astype(np.uint8)
                pred_m = postprocess_binary_mask(pred_m)
                inter = np.logical_and(pred_m > 0, gt > 0).sum()
                denom = pred_m.sum() + gt.sum() + 1e-8
                ds.append((2.0 * inter + 1e-8) / denom)
            md = float(np.mean(ds)) if ds else 0.0
            if md > best_dice:
                best_dice = md
                best_s = float(s)
                best_c = float(c)

    payload = {
        'score_threshold': 0.10 if best_dice < 0.02 else best_s,
        'color_threshold': 0.05 if best_dice < 0.02 else best_c,
        'objective': 'mean_validation_dice' if best_dice >= 0.02 else 'fallback_default_due_to_low_validation_dice',
        'objective_value': float(best_dice),
        'samples': len(cache),
        'model_input_height': IMG_HEIGHT,
        'model_input_width': IMG_WIDTH,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    with open(CALIBRATION_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print('\nDetection threshold calibration')
    print(f"Calibrated score_threshold: {payload['score_threshold']:.4f}")
    print(f"Calibrated color_threshold: {payload['color_threshold']:.4f}")
    print(f"Validation mean Dice (calibration objective): {best_dice:.4f}")
    print(f'Saved calibration: {CALIBRATION_OUTPUT}')


def main():
    spiked_images = _list_files(SPIKED_IMAGES_DIR)
    spiked_masks = _list_files(SPIKED_MASKS_DIR)
    clam_images = _list_files(CLAM_IMAGES_DIR)
    clam_masks = _list_files(CLAM_MASKS_DIR)

    spiked_pairs = _pair_by_stem(spiked_images, spiked_masks)
    clam_pairs = _pair_by_stem(clam_images, clam_masks)

    spiked_train, _ = _split_pairs(spiked_pairs, val_ratio=0.1)
    clam_train, clam_val = _split_pairs(clam_pairs, val_ratio=0.2)

    if len(spiked_train) < 1 or len(clam_train) < 1 or len(clam_val) < 1:
        raise ValueError('Not enough paired files after split. Ensure both domains have adequate pairs.')

    all_train_pairs = spiked_train + clam_train
    stats = compute_fg_stats(all_train_pairs)
    pos_weight = stats['suggested_pos_weight']
    POSITIVE_PIXEL_WEIGHT_VAR.assign(min(float(pos_weight), 150.0))

    print('\nMP-Set training setup')
    print(f'Spiked train pairs: {len(spiked_train)}')
    print(f'Clam train pairs: {len(clam_train)}')
    print(f'Clam val pairs: {len(clam_val)}')
    print(f"Detected inverted masks in train split: {stats['inverted_masks']}")
    print(f"Foreground ratio (mean/median): {stats['mean_fg']:.6f} / {stats['median_fg']:.6f}")
    print(f"Foreground ratio range: {stats['min_fg']:.6f} - {stats['max_fg']:.6f}")
    print(f"Masks with foreground < 0.1%: {stats['tiny_lt_0_1pct']}/{stats['count']}")
    print(f'Initial positive weight: {pos_weight:.2f}')
    print(f'Train patch mix (positive/hard-negative/random): {POSITIVE_PATCH_PROB:.2f}/{HARD_NEG_PATCH_PROB:.2f}/{1.0 - POSITIVE_PATCH_PROB - HARD_NEG_PATCH_PROB:.2f}')

    if stats['inverted_masks'] > 0:
        raise ValueError(
            'Detected inverted masks in spiked/clam subsets. Standardize both mask folders before training. '\
            'Run: python standardize_masks.py --masks-dir data/spiked_mask --in-place --backup-dir data/spiked_mask_backup '\
            'and python standardize_masks.py --masks-dir data/clam_mask --in-place --backup-dir data/clam_mask_backup'
        )

    steps_per_epoch = max(20, int(np.ceil(len(all_train_pairs) / BATCH_SIZE) * 4))
    val_steps = max(8, int(np.ceil(len(clam_val) / BATCH_SIZE) * 2))

    train_gen = DomainPatchGenerator(
        spiked_pairs=spiked_train,
        clam_pairs=clam_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=steps_per_epoch,
        training=True,
        clam_prob=CURRICULUM[0]['clam_prob'],
    )
    val_gen = DomainPatchGenerator(
        spiked_pairs=spiked_train,
        clam_pairs=clam_val,
        batch_size=BATCH_SIZE,
        steps_per_epoch=val_steps,
        training=False,
        clam_prob=1.0,
    )

    output_signature = (
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
    )

    train_ds = tf.data.Dataset.from_generator(lambda: iter(train_gen), output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(lambda: iter(val_gen), output_signature=output_signature)

    model = build_deep_resunet((IMG_HEIGHT, IMG_WIDTH, 3))
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=robust_hybrid_loss,
        metrics=[
            dice_coef,
            iou_coef,
            tf.keras.metrics.Precision(name='precision', thresholds=0.15),
            tf.keras.metrics.Recall(name='recall', thresholds=0.15),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.15),
            tf.keras.metrics.AUC(name='auc', from_logits=False),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR', from_logits=False, num_thresholds=200),
        ],
    )

    ckpt = callbacks.ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_pr_auc',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_pr_auc',
        mode='max',
        patience=16,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[
            CurriculumScheduler(train_gen=train_gen, initial_pos_weight=pos_weight),
            ckpt,
            reduce_lr,
            early_stop,
        ],
    )

    val_eval = model.evaluate(val_ds, steps=val_steps, verbose=0, return_dict=True)
    best_idx = int(np.argmax(history.history['val_pr_auc']))

    print('\nTraining summary')
    print(f'Best epoch: {best_idx + 1}')
    for name in ['val_dice_coef', 'val_iou_coef', 'val_precision', 'val_recall', 'val_pr_auc', 'val_loss']:
        if name in history.history:
            print(f'{name}: {history.history[name][best_idx]:.4f}')

    print('\nFinal clam validation metrics from model.evaluate():')
    for metric_name in ['loss', 'dice_coef', 'iou_coef', 'precision', 'recall', 'binary_accuracy', 'auc', 'pr_auc']:
        if metric_name in val_eval:
            print(f'  {metric_name}: {val_eval[metric_name]:.4f}')

    # Save training curves.
    dice = history.history.get('dice_coef', [])
    val_dice = history.history.get('val_dice_coef', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_run = len(loss)

    if epochs_run > 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        if dice:
            plt.plot(range(1, len(dice) + 1), dice, label='Train Dice')
        if val_dice:
            plt.plot(range(1, len(val_dice) + 1), val_dice, label='Val Dice')
        plt.legend()
        plt.title('Dice Coefficient')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(loss) + 1), loss, label='Train Loss')
        if val_loss:
            plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss')
        plt.legend()
        plt.title('Hybrid Loss')
        plt.savefig(TRAINING_PLOT)

    # Calibrate thresholds from best checkpoint when available.
    calibration_model = model
    if Path(MODEL_OUTPUT).is_file():
        calibration_model = tf.keras.models.load_model(
            MODEL_OUTPUT,
            custom_objects={
                'robust_hybrid_loss': robust_hybrid_loss,
                'dice_coef': dice_coef,
                'iou_coef': iou_coef,
            },
            compile=False,
        )
    calibrate_thresholds(calibration_model, clam_val)


if __name__ == '__main__':
    main()
