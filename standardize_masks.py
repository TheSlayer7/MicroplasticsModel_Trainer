import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
import tifffile

SUPPORTED_EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}


def load_mask(path):
    if path.suffix.lower() in {'.tif', '.tiff'}:
        mask = tifffile.imread(str(path))
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'Failed to read mask: {path}')
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def normalize_to_01(mask):
    if np.issubdtype(mask.dtype, np.integer):
        return mask.astype(np.float32) / float(np.iinfo(mask.dtype).max)
    return np.clip(mask.astype(np.float32), 0.0, 1.0)


def write_mask(path, mask_u8):
    if path.suffix.lower() in {'.tif', '.tiff'}:
        tifffile.imwrite(str(path), mask_u8)
    else:
        ok = cv2.imwrite(str(path), mask_u8)
        if not ok:
            raise ValueError(f'Failed to write mask: {path}')


def standardize_mask(path, in_place=False, output_dir=None, backup_dir=None):
    raw = load_mask(path)
    norm = normalize_to_01(raw)

    binary = (norm > 0.5).astype(np.uint8)
    was_inverted = float(binary.mean()) > 0.5
    if was_inverted:
        binary = 1 - binary

    standardized = (binary * 255).astype(np.uint8)

    if in_place:
        target = path
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
    else:
        if output_dir is None:
            raise ValueError('output_dir must be provided when not writing in place.')
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / path.name

    write_mask(target, standardized)

    fg_ratio = float(binary.mean())
    return {
        'path': str(path),
        'target': str(target),
        'inverted': was_inverted,
        'foreground_ratio': fg_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description='Standardize segmentation mask polarity and values.')
    parser.add_argument('--masks-dir', default='data/masks', help='Input masks directory.')
    parser.add_argument('--output-dir', default='data/masks_standardized', help='Output directory when not using --in-place.')
    parser.add_argument('--in-place', action='store_true', help='Overwrite masks in-place.')
    parser.add_argument('--backup-dir', default=None, help='Backup directory (recommended with --in-place).')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only; do not write files.')
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    if not masks_dir.is_dir():
        raise FileNotFoundError(f'Masks directory not found: {masks_dir}')

    mask_paths = sorted(
        p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not mask_paths:
        raise ValueError(f'No supported mask files found in: {masks_dir}')

    output_dir = None if args.in_place else Path(args.output_dir)
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    results = []
    for path in mask_paths:
        raw = load_mask(path)
        norm = normalize_to_01(raw)
        binary = (norm > 0.5).astype(np.uint8)
        would_invert = float(binary.mean()) > 0.5
        corrected = 1 - binary if would_invert else binary
        fg_ratio = float(corrected.mean())

        if args.dry_run:
            results.append({
                'path': str(path),
                'target': str(path if args.in_place else (output_dir / path.name)),
                'inverted': would_invert,
                'foreground_ratio': fg_ratio,
            })
            continue

        results.append(
            standardize_mask(
                path,
                in_place=args.in_place,
                output_dir=output_dir,
                backup_dir=backup_dir,
            )
        )

    inverted_count = sum(1 for r in results if r['inverted'])
    ratios = np.array([r['foreground_ratio'] for r in results], dtype=np.float64)

    print('Mask standardization summary')
    print(f'Total masks: {len(results)}')
    print(f'Inverted masks corrected: {inverted_count}')
    print(f'Foreground ratio mean/median: {float(np.mean(ratios)):.6f} / {float(np.median(ratios)):.6f}')
    print(f'Foreground ratio range: {float(np.min(ratios)):.6f} - {float(np.max(ratios)):.6f}')
    print(f'Dry run: {args.dry_run}')
    if args.in_place:
        print(f'In-place mode: True')
        print(f'Backup dir: {backup_dir if backup_dir else "(none)"}')
    else:
        print(f'Output dir: {output_dir}')


if __name__ == '__main__':
    main()
