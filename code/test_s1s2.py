import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from vlms import get_vlm
from utils.task import Task

COLORS = ['orange', 'blue', 'red', 'green', 'pink', 'purple', 'yellow', 'brown', 'black']
API_VLMS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gpt-4o',
    'claude-3.7',
    'gpt-o3',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--test_models', nargs="+", type=str, required=True)
    parser.add_argument('--max_cases', type=int, default=-1)
    parser.add_argument('--skip_runned', action='store_true')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--ic', action='store_true')
    parser.add_argument('--open_coord', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--force_points', action='store_true')
    parser.add_argument(
        '--which_s',
        type=str,
        default='both',
        choices=['s1', 's2', 'both'],
        help='Which split of S1/S2 to evaluate.',
    )
    parser.add_argument(
        '--s1s2_path',
        type=str,
        default='data/s1s2.json',
        help='Path to the S1/S2 JSON dataset.',
    )
    parser.add_argument(
        '--image_root',
        type=str,
        default='data/images_s1s2',
        help='Root directory for all images in s1s2.json.',
    )
    args = parser.parse_args()

    # tag for this run: {exp_name}_{MMDDHH}
    run_tag = f"{args.exp_name}_{datetime.now().strftime('%m%d%H')}"

    # load VLMs
    vlm_models = []
    for test_model in args.test_models:
        ModelClass, model_id = get_vlm(test_model)
        vlm_models.append(ModelClass(model=model_id))

    vlms = []
    for name, vlm in zip(args.test_models, vlm_models):
        print(f'Loading {vlm.__class__.__name__} as {name}')
        type_point = vlm.get_point_type()
        vlms.append((name, vlm, type_point))

    # init task env (for score + visualization only)
    task_env = Task(None)

    # load S1/S2 dataset
    print(f'Loading dataset from {args.s1s2_path} ...')
    with open(args.s1s2_path, 'r') as f:
        s1s2_data = json.load(f)

    # filter by S1/S2 if needed
    if args.which_s in ['s1', 's2']:
        s1s2_data = [s for s in s1s2_data if s.get('s1_or_s2') == args.which_s]
        print(f'Filtered to {len(s1s2_data)} samples for split {args.which_s}')
    else:
        print(f'Total samples: {len(s1s2_data)} (S1 + S2)')

    processed_count = 0

    for idx, sample in tqdm(enumerate(s1s2_data), total=len(s1s2_data)):
        if args.max_cases > 0 and processed_count >= args.max_cases:
            break

        polygon = sample['polygon']
        image_rel_path = sample['image_path']  # stored in json
        image_path = os.path.join(args.image_root, image_rel_path)
        height, width = sample['height'], sample['width']
        lang = sample['lang']
        s1_or_s2 = sample['s1_or_s2']   # 's1' or 's2'
        subclass = sample['subclasses']   # e.g., recommendation, etc.

        # simple numeric tag: 1, 2, 3, ...
        image_tag = str(idx + 1)

        # directory structure: results/s?/{exp_name}_{MMDDHH}/{tag}/
        s_folder = 's1' if s1_or_s2 == 's1' else 's2'
        base_dir = os.path.join(args.save_path, s_folder, run_tag, image_tag)

        info_path = os.path.join(base_dir, 'info.npy')
        vis_path = base_dir  # Task.visualize will save vis.png under this dir

        if args.skip_runned and os.path.exists(info_path):
            print(f'Skipping already processed sample: {info_path}')
            continue

        os.makedirs(base_dir, exist_ok=True)

        print(f'\n[Sample #{idx+1}] tag={image_tag}  prompt="{lang}"')

        pkg = {}                  # for visualization
        results_per_task = []     # list of dicts for info.npy
        any_success = False

        for i, (vlm_name, vlm, type_point) in enumerate(vlms):
            question = lang

            # call VLM
            if args.debug_mode:
                try:
                    ans = vlm(question, image_path) if not args.ic else vlm.ic_call(question, image_path)
                except Exception as e:
                    print(f'[{vlm_name}] Error in debug_mode: {e}')
                    ans = None
            else:
                ans = None
                for loop_n in range(5):
                    try:
                        ans = vlm(question, image_path) if not args.ic else vlm.ic_call(question, image_path)
                        if vlm_name in API_VLMS:
                            time.sleep(10)  # throttle API-based models
                        break
                    except Exception as e:
                        print(f'[{vlm_name}] Failed to get answer {loop_n+1}/5: {e}')
                        print('Waiting 5 seconds before retry...')
                        time.sleep(5)
                        ans = None

            if ans is None:
                print(f'[{vlm_name}] Failed to get answer, skipping this model.')
                continue

            any_success = True

            # scoring
            score = task_env.score(height, width, polygon, ans, type=type_point)

            # pack info for visualization (one color per model)
            pkg[vlm_name] = {
                'type': type_point,
                'points': ans,
                'color': COLORS[i % len(COLORS)],
            }

            # info to save
            save_info = {
                'ans': ans,
                'score': score,
                's1_or_s2': s1_or_s2,
                'subclass': subclass,
                'question': question,
                'image_path': image_path,
                'image_rel_path': image_rel_path,
                'lang': lang,
                'vlm_name': vlm_name,
                'type_point': type_point,
                'height': height,
                'width': width,
                'polygon': polygon,
                'sample_idx': idx,
                'tag': image_tag,
            }
            results_per_task.append(save_info)

        if not any_success:
            print('No successful VLM outputs for this sample, skipping visualization.')
            continue

        # save all model infos for this sample in one info.npy
        np.save(info_path, results_per_task, allow_pickle=True)
        print(f'Saved info to {info_path}')

        # visualization (Task.visualize should write something like {vis_path}/vis.png)
        task_env.visualize(image_path, polygon, pkg, lang, vis_path)
        print(f'Visualizing to {vis_path}')

        processed_count += 1

    print(f'\nDone. Processed {processed_count} samples.')


if __name__ == '__main__':
    main()
