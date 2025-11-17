#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np


def iou(bbx1, bbx2):
    """
    IoU between two axis-aligned bounding boxes: [x1, y1, x2, y2].
    """
    x1_min, y1_min, x1_max, y1_max = bbx1
    x2_min, y2_min, x2_max, y2_max = bbx2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_from_polygon(polygon):
    """
    Take COCO-style polygon [[x1, y1, ..., xN, yN]] and return GT bbox in pixels.
    """
    poly = np.array(polygon[0]).reshape(-1, 2)
    x_min = float(np.min(poly[:, 0]))
    y_min = float(np.min(poly[:, 1]))
    x_max = float(np.max(poly[:, 0]))
    y_max = float(np.max(poly[:, 1]))
    return [x_min, y_min, x_max, y_max]


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))
    # ensure non-negative size
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return [x1, y1, x2, y2]


def compute_iou_normalized(info, human_error=0.0):
    """
    IoU-normalized metric:

      score_raw     = IoU(predicted_bbox, GT_bbox_from_polygon)
      human_score   = IoU(enlarged_GT_bbox, GT_bbox_from_polygon)
      score_norm    = score_raw / human_score   (if human_score > 0)

    Returns score_norm in [0, +inf), but typically <= 1.0 for reasonable models.
    """
    if info['type_point'] != 'bbx':
        # For S1/S2 we mainly care about bbox; if points show up, just return stored score.
        return float(info['score'])

    h, w = int(info['height']), int(info['width'])
    polygon = info['polygon']
    pred_norm = info['ans']  # [x1, y1, x2, y2] in normalized coords

    # GT bbox from polygon (pixel coords)
    gt_box = bbox_from_polygon(polygon)

    # Pred bbox in pixel coords (clamped)
    x1 = pred_norm[0] * w
    y1 = pred_norm[1] * h
    x2 = pred_norm[2] * w
    y2 = pred_norm[3] * h
    pred_box = clamp_box([x1, y1, x2, y2], w, h)

    raw_iou = iou(pred_box, gt_box)

    # Human baseline: slightly enlarged GT box in normalized coords
    x_min, y_min, x_max, y_max = gt_box
    gt_norm = [x_min / w, y_min / h, x_max / w, y_max / h]
    human_box_norm = [
        gt_norm[0] - human_error,
        gt_norm[1] - human_error,
        gt_norm[2] + human_error,
        gt_norm[3] + human_error,
    ]
    # Convert to pixel, clamp
    hx1 = human_box_norm[0] * w
    hy1 = human_box_norm[1] * h
    hx2 = human_box_norm[2] * w
    hy2 = human_box_norm[3] * h
    human_box = clamp_box([hx1, hy1, hx2, hy2], w, h)

    human_iou = iou(human_box, gt_box)
    if human_iou <= 0:
        return 0.0

    return raw_iou / human_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path',
        type=str,
        required=True,
        help='Root folder that contains s1/ and s2/ result dirs from test_s1s2.py',
    )
    args = parser.parse_args()
    root = args.result_path

    # We expect:
    #   {root}/s1/{run_tag}/{tag}/info.npy
    #   {root}/s2/{run_tag}/{tag}/info.npy
    paths = {
        's1': glob.glob(os.path.join(root, 's1', '*', '*', 'info.npy')),
        's2': glob.glob(os.path.join(root, 's2', '*', '*', 'info.npy')),
    }

    if not paths['s1'] and not paths['s2']:
        print(f'No info.npy found under {root}/s1 or {root}/s2. Check result_path.')
        return

    # stats[model] = dict with sums and counts
    stats = {}

    for split in ['s1', 's2']:
        for info_path in paths[split]:
            arr = np.load(info_path, allow_pickle=True)
            # Each info.npy from test_s1s2.py is a list/array of dicts (one per model)
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    records = arr.tolist()
                else:
                    records = [arr.item()]
            elif isinstance(arr, dict):
                records = [arr]
            else:
                continue

            for info in records:
                model = info['vlm_name']
                s1_or_s2 = info['s1_or_s2']  # 's1' or 's2'

                if model not in stats:
                    stats[model] = {
                        's1_sum': 0.0,
                        's1_count': 0,
                        's2_sum': 0.0,
                        's2_count': 0,
                    }

                score_norm = compute_iou_normalized(info)

                if s1_or_s2 == 's1':
                    stats[model]['s1_sum'] += score_norm
                    stats[model]['s1_count'] += 1
                else:
                    stats[model]['s2_sum'] += score_norm
                    stats[model]['s2_count'] += 1

    # Print results
    print('\n==== S1/S2 IoU-normalized scores ====\n')
    for model, v in sorted(stats.items()):
        s1_n = v['s1_count']
        s2_n = v['s2_count']
        s1_avg = v['s1_sum'] / s1_n if s1_n > 0 else 0.0
        s2_avg = v['s2_sum'] / s2_n if s2_n > 0 else 0.0
        all_sum = v['s1_sum'] + v['s2_sum']
        all_n = s1_n + s2_n
        all_avg = all_sum / all_n if all_n > 0 else 0.0

        print(f'Model: {model}')
        print(f'  S1: {s1_avg:.4f}  (N={s1_n})')
        print(f'  S2: {s2_avg:.4f}  (N={s2_n})')
        print(f'  All: {all_avg:.4f} (N={all_n})')
        print('')

    print('Done.')


if __name__ == '__main__':
    main()
