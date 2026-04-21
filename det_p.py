import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile

# Reduce OpenCV TIFF metadata warning noise.
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
if hasattr(cv2, 'setLogLevel'):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

MODEL_CANDIDATES = ['my_custom_resunet.keras', 'myModel.keras']
CALIBRATION_FILE = 'detection_calibration.json'

# Conservative defaults for clam-style noisy samples.
DEFAULT_SCORE_THRESHOLD = 0.42
DEFAULT_COLOR_THRESHOLD = 0.05


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def tversky_coef(y_true, y_pred, alpha=0.10, beta=0.90, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def weighted_bce_loss(y_true, y_pred, pos_weight=800.0):
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    bce = -(pos_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


def robust_hybrid_loss(y_true, y_pred):
    return 0.50 * weighted_bce_loss(y_true, y_pred) + 0.30 * (1.0 - tversky_coef(y_true, y_pred)) + 0.20 * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


CUSTOM_OBJECTS = {
    'robust_hybrid_loss': robust_hybrid_loss,
    'dice_coef': dice_coef,
    'iou_coef': iou_coef,
}


def resolve_model_path(model_path=None):
    if model_path:
        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(f'Model file not found: {p}')
        return str(p)

    for candidate in MODEL_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    raise FileNotFoundError(f'No model file found. Tried: {", ".join(MODEL_CANDIDATES)}')


def load_model_for_inference(model_path=None):
    resolved = resolve_model_path(model_path)
    model = tf.keras.models.load_model(resolved, custom_objects=CUSTOM_OBJECTS, compile=False)
    return model, resolved


def load_detection_calibration(calibration_path):
    path = Path(calibration_path)
    if not path.is_file():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if 'score_threshold' not in data or 'color_threshold' not in data:
        return None

    try:
        score_t = float(data['score_threshold'])
        color_t = float(data['color_threshold'])
    except (TypeError, ValueError):
        return None

    if not np.isfinite(score_t) or not np.isfinite(color_t):
        return None

    objective_value = data.get('objective_value', None)
    if objective_value is not None:
        try:
            objective_value = float(objective_value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(objective_value) or objective_value < 0.02:
            return None

    # Guard against permissive legacy values.
    if score_t < 0.28 or color_t < 0.01:
        return None

    return data


def infer_domain_from_path(image_path):
    s = str(image_path).lower()
    if 'spiked' in s:
        return 'spiked'
    if 'clam' in s:
        return 'clam'
    return 'clam'


def domain_profile(domain):
    if domain == 'spiked':
        return {
            'score_floor': 0.34,
            'color_floor': 0.03,
            'min_area': 4.0,
            'max_area_ratio': 0.20,
            'blend_model': 0.98,
            'blend_color': 0.02,
        }
    return {
        'score_floor': 0.42,
        'color_floor': 0.05,
        'min_area': 6.0,
        'max_area_ratio': 0.16,
        'blend_model': 0.97,
        'blend_color': 0.03,
    }


def get_model_input_hw(model, fallback=(512, 512)):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) < 4 or shape[1] is None or shape[2] is None:
        return fallback
    return int(shape[1]), int(shape[2])


def read_image_rgb_float(image_path):
    p = Path(image_path)
    if p.suffix.lower() in {'.tif', '.tiff'}:
        arr = tifffile.imread(str(p))
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f'OpenCV failed to read image: {image_path}')

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


def load_mask_binary(mask_path, target_hw):
    p = Path(mask_path)
    if p.suffix.lower() in {'.tif', '.tiff'}:
        mask = tifffile.imread(str(p))
    else:
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'OpenCV failed to read mask: {mask_path}')
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    if np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.float32) / float(np.iinfo(mask.dtype).max)
    else:
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return (mask > 0.5).astype(np.uint8)


def warm_spot_map(img):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    warm = np.maximum(0.0, r - 0.55 * g - 0.45 * b) + 0.6 * np.maximum(0.0, g - 0.75 * b)
    return np.clip(warm, 0.0, 1.0)


def preprocess_for_model(img):
    warm = warm_spot_map(img)
    warm_c = np.expand_dims(warm, axis=-1)
    out = img.copy()
    out[..., 0:1] = np.clip(out[..., 0:1] + 0.30 * warm_c, 0.0, 1.0)
    out[..., 1:2] = np.clip(out[..., 1:2] + 0.15 * warm_c, 0.0, 1.0)
    out[..., 2:3] = np.clip(0.90 * out[..., 2:3], 0.0, 1.0)
    return out


def predict_tta(model, input_img, tta_views=2):
    if tta_views <= 1:
        views = [input_img]
    elif tta_views == 2:
        views = [input_img, np.fliplr(input_img)]
    else:
        views = [
            input_img,
            np.fliplr(input_img),
            np.flipud(input_img),
            np.flipud(np.fliplr(input_img)),
        ]

    preds = []
    for i, view in enumerate(views):
        pred = model.predict(np.expand_dims(view, axis=0), verbose=0)[0]
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = np.squeeze(pred, axis=-1)
        pred = np.clip(np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if tta_views >= 2 and i == 1:
            pred = np.fliplr(pred)
        elif tta_views == 4 and i == 2:
            pred = np.flipud(pred)
        elif tta_views == 4 and i == 3:
            pred = np.fliplr(np.flipud(pred))

        preds.append(pred)

    return np.mean(np.stack(preds, axis=0), axis=0)


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def estimate_presence(pred_map, color_map, mask_ratio, n_boxes):
    q995 = float(np.quantile(pred_map, 0.995))
    cq995 = float(np.quantile(color_map, 0.995))
    area_signal = min(1.0, mask_ratio * 55.0)
    box_signal = min(1.0, n_boxes / 18.0)
    return float(np.clip(0.52 * q995 + 0.26 * cq995 + 0.14 * area_signal + 0.08 * box_signal, 0.0, 1.0))


def detect_and_draw(
    image_path,
    model,
    domain,
    out_path='detected_boxes.png',
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    color_threshold=DEFAULT_COLOR_THRESHOLD,
    tta_views=2,
    max_area_ratio=0.20,
    max_boxes=500,
):
    profile = domain_profile(domain)
    score_threshold = max(float(score_threshold), profile['score_floor'], 0.15)
    color_threshold = max(float(color_threshold), profile['color_floor'])
    max_area_ratio = min(max_area_ratio, profile['max_area_ratio'])

    original = read_image_rgb_float(image_path)
    h0, w0 = original.shape[:2]

    in_h, in_w = get_model_input_hw(model)
    model_input = preprocess_for_model(original)
    resized = cv2.resize(model_input, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    pred = predict_tta(model, resized, tta_views=tta_views)
    pred = cv2.resize(pred.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    color = warm_spot_map(original).astype(np.float32)
    color = np.clip(np.nan_to_num(color, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    pred_blur = cv2.GaussianBlur(pred, (0, 0), sigmaX=0.5)
    color_blur = cv2.GaussianBlur(color, (0, 0), sigmaX=0.5)
    combined = np.clip(profile['blend_model'] * pred_blur + profile['blend_color'] * color_blur, 0.0, 1.0)

    adaptive_score = max(score_threshold, float(np.quantile(pred_blur, 0.985)) * 0.62)
    adaptive_color = max(color_threshold, float(np.quantile(color_blur, 0.985)) * 0.28)
    high_thr = adaptive_score
    low_thr = max(score_threshold * 0.80, adaptive_score * 0.80)

    binary = (pred_blur > high_thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    h, w = binary.shape
    image_area = float(h * w)
    min_area = max(profile['min_area'], image_area * 0.000010)

    # Remove tiny residual speckles before scoring/statistics.
    n_labels_pre, labels_pre, stats_pre, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lbl in range(1, n_labels_pre):
        if int(stats_pre[lbl, cv2.CC_STAT_AREA]) >= int(min_area):
            cleaned[labels_pre == lbl] = 1
    binary = cleaned

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []

    for lbl in range(1, n_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area <= min_area or area >= image_area * max_area_ratio:
            continue

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        region = labels == lbl
        mean_model = float(pred_blur[region].mean()) if np.any(region) else 0.0
        max_model = float(pred_blur[region].max()) if np.any(region) else 0.0
        mean_comb = float(combined[region].mean()) if np.any(region) else 0.0
        mean_color = float(color[region].mean()) if np.any(region) else 0.0

        if max_model < max(low_thr * 0.90, score_threshold * 0.85):
            continue
        if mean_model < low_thr * 0.55 and mean_color < adaptive_color * 0.80:
            continue

        rank = 0.62 * mean_model + 0.23 * max_model + 0.10 * mean_comb + 0.05 * mean_color
        candidates.append((rank, x, y, bw, bh, area, mean_model, max_model, mean_color))

    candidates.sort(key=lambda z: z[0], reverse=True)
    if candidates:
        nms_boxes = [[c[1], c[2], c[3], c[4]] for c in candidates]
        nms_scores = [float(c[0]) for c in candidates]
        keep = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=0.0, nms_threshold=0.20)
        if len(keep) > 0:
            keep = np.array(keep).flatten().tolist()
            candidates = [candidates[k] for k in keep]

    selected = candidates[:max_boxes]
    display = (original * 255.0).astype(np.uint8)
    boxes = []

    for _, x, y, bw, bh, area, mean_model, max_model, mean_color in selected:
        conf = float(np.clip(0.75 * max_model + 0.25 * mean_color, 0.0, 1.0))
        boxes.append((x, y, bw, bh, area, mean_model, max_model, mean_color, conf))
        cv2.rectangle(display, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        label = f'Plastic {conf * 100:.1f}%'
        cv2.putText(display, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(display)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    model_positive_rate = float((pred_blur > 0.10).mean())
    disagreement = float(np.mean(np.abs(pred_blur - color_blur)))
    likelihood = estimate_presence(pred_blur, color_blur, float(binary.mean()), len(boxes))

    stats = {
        'domain': domain,
        'num_boxes': len(boxes),
        'num_candidates': len(candidates),
        'hit_max_boxes_cap': bool(len(candidates) > max_boxes),
        'mask_ratio': float(binary.mean()),
        'mean_prediction_score': float(pred.mean()),
        'mean_color_prior': float(color.mean()),
        'adaptive_score_threshold': float(adaptive_score),
        'adaptive_color_threshold': float(adaptive_color),
        'high_score_threshold': float(high_thr),
        'low_score_threshold': float(low_thr),
        'base_score_threshold': float(score_threshold),
        'base_color_threshold': float(color_threshold),
        'q995_model': float(np.quantile(pred_blur, 0.995)),
        'q999_model': float(np.quantile(pred_blur, 0.999)),
        'model_positive_rate_t01': model_positive_rate,
        'model_color_disagreement': disagreement,
        'plastic_presence_likelihood': float(likelihood),
        'avg_box_confidence': float(np.mean([b[-1] for b in boxes])) if boxes else 0.0,
        'max_box_confidence': float(np.max([b[-1] for b in boxes])) if boxes else 0.0,
        'tta_views': int(tta_views),
        'model_input_height': int(in_h),
        'model_input_width': int(in_w),
    }

    return binary, stats


def main():
    parser = argparse.ArgumentParser(description='Detect microplastics from microscopy images.')
    parser.add_argument('--image', required=True, help='Path to input image.')
    parser.add_argument('--model', default=None, help='Optional model path.')
    parser.add_argument('--mask', default=None, help='Optional GT mask for metrics.')
    parser.add_argument('--domain', choices=['auto', 'clam', 'spiked'], default='auto', help='Domain profile for thresholds and filtering.')
    parser.add_argument('--calibration', default=CALIBRATION_FILE, help='Calibration JSON path.')
    parser.add_argument('--no-calibration', action='store_true', help='Ignore calibration file.')
    parser.add_argument('--score-threshold', type=float, default=None, help='Override base score threshold.')
    parser.add_argument('--color-threshold', type=float, default=None, help='Override base color threshold.')
    parser.add_argument('--max-area-ratio', type=float, default=0.20, help='Maximum allowed component area ratio.')
    parser.add_argument('--max-boxes', type=int, default=500, help='Maximum number of boxes to draw.')
    parser.add_argument('--tta-views', type=int, choices=[1, 2, 4], default=2, help='Number of TTA views (1, 2, or 4).')
    parser.add_argument('--output', default='detected_boxes.png', help='Output image path.')
    args = parser.parse_args()

    try:
        model, model_path = load_model_for_inference(args.model)

        domain = infer_domain_from_path(args.image) if args.domain == 'auto' else args.domain
        base_score = DEFAULT_SCORE_THRESHOLD
        base_color = DEFAULT_COLOR_THRESHOLD
        threshold_source = 'builtin_default'

        if not args.no_calibration:
            cal = load_detection_calibration(args.calibration)
            if cal is not None:
                base_score = float(cal['score_threshold'])
                base_color = float(cal['color_threshold'])
                threshold_source = args.calibration

        if args.score_threshold is not None:
            base_score = float(args.score_threshold)
            threshold_source = 'manual_cli'
        if args.color_threshold is not None:
            base_color = float(args.color_threshold)
            threshold_source = 'manual_cli'

        if not np.isfinite(base_score) or not np.isfinite(base_color):
            base_score = DEFAULT_SCORE_THRESHOLD
            base_color = DEFAULT_COLOR_THRESHOLD
            threshold_source = 'builtin_default'

        pred_mask, stats = detect_and_draw(
            image_path=args.image,
            model=model,
            domain=domain,
            out_path=args.output,
            score_threshold=base_score,
            color_threshold=base_color,
            tta_views=int(args.tta_views),
            max_area_ratio=float(args.max_area_ratio),
            max_boxes=int(args.max_boxes),
        )

        print(f'Loaded model: {model_path}')
        print(f'Domain profile: {domain}')
        print(f'Threshold source: {threshold_source}')
        print(f'Saved output: {args.output}')

        print('\nDetection summary')
        print(f"Model input size used: {stats['model_input_height']}x{stats['model_input_width']}")
        print(f"Boxes: {stats['num_boxes']}")
        print(f"Candidates before cap: {stats['num_candidates']}")
        print(f"Hit max box cap: {stats['hit_max_boxes_cap']}")
        print(f"Mask coverage: {stats['mask_ratio'] * 100:.4f}%")
        print(f"Mean model score: {stats['mean_prediction_score']:.4f}")
        print(f"Mean warm-color prior: {stats['mean_color_prior']:.4f}")
        print(f"Adaptive score threshold: {stats['adaptive_score_threshold']:.4f}")
        print(f"Adaptive color threshold: {stats['adaptive_color_threshold']:.4f}")
        print(f"High/Low score thresholds: {stats['high_score_threshold']:.4f} / {stats['low_score_threshold']:.4f}")
        print(f"Base score/color thresholds used: {stats['base_score_threshold']:.4f} / {stats['base_color_threshold']:.4f}")
        print(f"Model q99.5/q99.9: {stats['q995_model']:.4f} / {stats['q999_model']:.4f}")
        print(f"Model positive rate @0.1: {stats['model_positive_rate_t01'] * 100:.4f}%")
        print(f"Model-color disagreement: {stats['model_color_disagreement']:.4f}")
        print(f"TTA views used: {stats['tta_views']}")
        print(f"Plastic presence likelihood: {stats['plastic_presence_likelihood'] * 100:.2f}%")
        print(f"Avg/Max box confidence: {stats['avg_box_confidence'] * 100:.2f}% / {stats['max_box_confidence'] * 100:.2f}%")

        if args.mask:
            gt = load_mask_binary(args.mask, pred_mask.shape)
            m = compute_metrics(pred_mask, gt)
            print('\nAccuracy metrics (vs provided mask)')
            print(f"IoU: {m['iou']:.4f}")
            print(f"Dice/F1: {m['dice']:.4f}")
            print(f"Precision: {m['precision']:.4f}")
            print(f"Recall: {m['recall']:.4f}")
            print(f"Pixel Accuracy: {m['accuracy']:.4f}")
            print(f"TP/FP/FN/TN: {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}")

    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f'Error: {exc}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
