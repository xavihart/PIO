import argparse
from vlms import get_vlm
from utils.task import Task  # not strictly needed here, but keeping for consistency
from tqdm import tqdm
import os
import time
import numpy as np
import json
import re
import cv2
from datetime import datetime


def visualize_trajectory(image_path, save_path, trajectory, title="Trajectory"):
    """
    Draws the trajectory on the image with a title and saves it.

    Args:
        image_path (str): Path to the image.
        save_path (str): Path to save the output image.
        trajectory: list-like of points with .x and .y (float in [0, 1]).
        title (str): Title to be drawn on the image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    h, w = image.shape[:2]

    # Add padding for title
    title_height = 40
    image = cv2.copyMakeBorder(
        image,
        title_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    # Draw title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = title_height - 10
    cv2.putText(
        image,
        title,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    # Draw trajectory
    points = trajectory
    num_points = len(points)
    if num_points < 2:
        raise ValueError("Trajectory must contain at least two points to draw lines.")

    for i in range(num_points - 1):
        # Color from blue to red
        ratio = i / (num_points - 1)
        b = int(255 * (1 - ratio))
        r = int(255 * ratio)
        color = (b, 0, r)  # BGR for OpenCV
        pt1 = [
            int(points[i].x * w),
            int(points[i].y * h) + title_height,
        ]
        pt2 = [
            int(points[i + 1].x * w),
            int(points[i + 1].y * h) + title_height,
        ]
        cv2.line(image, pt1, pt2, color, thickness=2)

    # Save output
    cv2.imwrite(save_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='data/s3.json',
                        help='Path to the S3 JSON dataset (list of dicts).')
    parser.add_argument('--image_root', type=str, default='data/images_s3',
                        help='Root directory for images referenced in s3.json.')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--test_model', type=str, default='gpt-4o')
    parser.add_argument('--max_cases', type=int, default=-1)
    parser.add_argument('--skip_runned', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()

    # Tag for this run: {exp_name}_{MMDDHH}
    run_tag = f"{args.exp_name}_{datetime.now().strftime('%m%d%H')}"

    # Load VLM
    test_model = args.test_model
    ModelClass, model_id = get_vlm(test_model)
    vlm_model = ModelClass(model=model_id)
    print(f'Loaded {vlm_model.__class__.__name__} as {test_model}')

    # Load dataset (list of dicts)
    with open(args.json_path, 'r') as f:
        s3_data = json.load(f)
    print(f'Loaded {len(s3_data)} S3 samples from {args.json_path}')

    processed_count = 0

    for idx, sample in tqdm(enumerate(s3_data), total=len(s3_data)):
        if args.max_cases > 0 and processed_count >= args.max_cases:
            break

        # New format: each sample is a dict with at least image_path and lang
        image_rel_path = sample['image_path']
        question = sample['lang']

        image_path = os.path.join(args.image_root, image_rel_path)

        # Simple numeric tag: 1, 2, 3, ...
        tag = str(idx + 1)

        # Directory: results/s3/{exp_name}_{MMDDHH}/{tag}/
        base_dir = os.path.join(args.save_path, 's3', run_tag, tag)
        os.makedirs(base_dir, exist_ok=True)

        info_path = os.path.join(base_dir, 'info.npy')
        vis_path = os.path.join(base_dir, 'vis.png')

        if args.skip_runned and os.path.exists(info_path):
            print(f'Skipping already processed sample tag={tag} (info.npy exists)')
            continue

        print(f'\n[Sample #{idx+1}] tag={tag}')
        print(f'  Image: {image_path}')
        print(f'  Prompt: {question}')

        # Extract (x, y) from filename: ..._x_y.jpg or .png
        match = re.search(r"_([0-9]*\.?[0-9]+)_([0-9]*\.?[0-9]+)\.(jpg|png)", image_path)
        if not match:
            print(f'  Could not parse coordinates from: {image_path}, skipping.')
            continue

        x = float(match.group(1))
        y = float(match.group(2))

        if test_model == 'molmo':
            x = x * 100
            y = y * 100

        # Call the model
        success = False
        if not args.debug_mode:
            for _ in range(5):
                try:
                    pred_trajectory = vlm_model.s3(question, f"({x},{y})", image_path)
                    success = True
                    break
                except Exception as e:
                    print(f'  Error during VLM call: {e}, retrying...')
                    time.sleep(2)
        else:
            # single shot in debug
            pred_trajectory = vlm_model.s3(question, f"({x},{y})", image_path)
            success = True

        if not success:
            print('  Failed all retries, skipping this sample.')
            continue

        print(f'  Got trajectory with {len(pred_trajectory)} points.')

        # Visualize trajectory
        visualize_trajectory(image_path, vis_path, pred_trajectory, title=question)
        print(f'  Saved visualization to {vis_path}')

        # Save trajectory (info)
        np.save(info_path, pred_trajectory, allow_pickle=True)
        print(f'  Saved info to {info_path}')

        processed_count += 1

    print(f'\nDone. Processed {processed_count} samples.')


if __name__ == '__main__':
    main()

"""
python code/test_s3.py \
  --test_model gemini-2.5-pro \
  --max_cases 5 \
  --exp_name demo_gemini_s3 \
  --json_path data/s3.json \
  --image_root data/images_s3 \
  --save_path results
"""