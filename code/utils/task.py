import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image


class Task(object):
    def __init__(self, path):
        if path is not None:
            self.task_path = path
            self.annotation_path = path + '_annotations.coco.json'
        self.annotations = None
        # self.load_task()
    

    @staticmethod
    def polygon_to_mask(segmentation, width, height):
        """
        Converts COCO-style polygon segmentation into a binary mask.

        Args:
            segmentation (list): COCO-style segmentation polygon [[x1, y1, x2, y2, ..., xn, yn]]
            width (int): width of the image
            height (int): height of the image

        Returns:
            numpy.ndarray: Binary mask (height, width) where pixels inside the polygon are 1, else 0
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # reshape the polygon points into (N, 2)
        
        poly = [np.array(poly).reshape((-1, 2)).astype(np.int32) for poly in segmentation]

        # fill the polygon
        cv2.fillPoly(mask, poly, 255)

        return mask
    
    def visualize_polygon(self, image_path, polygon, save_path, lang):
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_rgb)

        # Plot ground truth polygon
        try:
            polygon_np = np.array(polygon).reshape(-1, 2)
        except:
            polygon_np = np.array(polygon[0]).reshape(-1, 2)
        
        polygon_patch = patches.Polygon(polygon_np, closed=True, linewidth=2, edgecolor='lime', facecolor='none', label='Ground Truth')
        ax.add_patch(polygon_patch)

        # Add language description
        ax.set_title(lang, fontsize=14)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        
        
    def visualize(self, image_path, polygon, points_dict, lang, save_path, transparent=True):
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # make it transparent
        if transparent:
            overlay = image_rgb.copy()
            image_rgb = cv2.addWeighted(overlay, 0.8, np.full_like(overlay, 255), 0.2, 0)

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_rgb)

        # Plot ground truth polygon
        try:
            polygon_np = np.array(polygon).reshape(-1, 2)
        except:
            polygon_np = np.array(polygon[0]).reshape(-1, 2)
        
        polygon_patch = patches.Polygon(polygon_np, closed=True, linewidth=2, edgecolor='lime', facecolor='none', label='Ground Truth')
        ax.add_patch(polygon_patch)

        # Plot predictions from points_dict
        legend_handles = [polygon_patch]
        for model_name, model_data in points_dict.items():
            color = model_data.get('color', 'yellow')
            points = model_data['points']

            if model_data['type'] == 'bbx':
                x1, y1, x2, y2 = points
                # rescale to h, w
                x1, x2 = x1 * image.shape[1], x2 * image.shape[1]
                y1, y2 = y1 * image.shape[0], y2 * image.shape[0]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                        edgecolor=color, facecolor='none', label=model_name)
                ax.add_patch(rect)
                # also visualize the center point of the bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ax.plot(cx, cy, marker='o', color=color, markersize=20, markeredgecolor="white")
                legend_handles.append(rect)
            elif model_data['type'] == 'points':
                xs, ys = zip(*points)
                # rescale to h, w
                xs = [x * image.shape[1] for x in xs]
                ys = [y * image.shape[0] for y in ys]
                scatter = ax.scatter(xs, ys, c=color, s=70, label=model_name, marker='x', linewidths=5, edgecolors='black')
                legend_handles.append(scatter)

        # Add language description
        ax.set_title(lang, fontsize=14)

        # Set legend to the right
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.axis('off')
        plt.tight_layout()
        # if save_path is a directory, save the image with the name of the image
        if os.path.isdir(save_path):
            plt.savefig(save_path + '/results.png')
        else:
            plt.savefig(save_path)
        
    def score(self, h, w, polygon, points, type='bbx'):
        """
        points:
           if type == 'points':
                points = [[x1, y1], [x2, y2], ..., [xn, yn]]
           if type == 'bbx':
                points = [x1, y1, x2, y2]
        """
        assert type in ['points', 'bbx']
        
        # normalize points to the image size, current 0-1
        if type == 'points':
            points = [[int(p[0] * w), int(p[1] * h)] for p in points]
        elif type == 'bbx': 
            points = [int(points[0] * w), int(points[1] * h), int(points[2] * w), int(points[3] * h)]
        
        if type == 'points':
            mask = self.polygon_to_mask(polygon, w, h)
            avg = 0
            for point in points:
                if mask[point[1], point[0]] != 0:
                    avg += 1
            return avg / len(points)
        elif type == 'bbx':
            # sample in bbx
            mask = self.polygon_to_mask(polygon, w, h)
            avg = 0
            # sample uniformly in bbx
            x1, y1, x2, y2 = points
            for i in range(x1, x2):
                for j in range(y1, y2):
                    if mask[j, i] != 0:
                        avg += 1
            return avg / ((x2 - x1 + 1) * (y2 - y1 + 1))
            
    def get_task(self, task_id):
        task = self.annotations[task_id]
        return task['polygon'], task['height'], task['width'], task['image_path'], task['lang']
    
    
    def load_task_combined(self, combined_path='data/combined_dataset_polished.json'):
        with open(combined_path) as f:
            data = json.load(f)
            print(data[0].keys())
            self.annotations = data
                
                
        # sort the keys and save into a list
        # self.annotations = []
        # for key in sorted(annotations.keys()):
        #     self.annotations.append(annotations[key])
    
    
    def load_task(self, ):
        
        with open(self.annotation_path) as f:
            data = json.load(f)
            print(data.keys())
            
            langs = {}
            for lang in data['categories']:
                langs[lang['id']] = lang['name']
            
            images = {}
            for image in data['images']:
                images[image['id']] = {}
                images[image['id']]['file_name'] = image['file_name']
                images[image['id']]['height'] = image['height']
                images[image['id']]['width'] = image['width']
            
            annotations = {}
            
            for annotation in data['annotations']:
                image_id = annotation['image_id']
                lang_id = annotation['category_id']
                polygon = annotation['segmentation']
                
                if (image_id, lang_id) not in annotations:
                    annotations[(image_id, lang_id)] = {
                        'polygon': [],
                        'image_path': None,
                        'height': None,
                        'width': None,
                        'lang': None
                    }
        
                annotations[(image_id, lang_id)]['polygon'].append(polygon[0])
                annotations[(image_id, lang_id)]['image_path'] = images[image_id]['file_name']
                annotations[(image_id, lang_id)]['height'] = images[image_id]['height']
                annotations[(image_id, lang_id)]['width'] = images[image_id]['width']
                annotations[(image_id, lang_id)]['lang'] = langs[lang_id]
            
            # sort the keys and save into a list
            self.annotations = []
            for key in sorted(annotations.keys()):
                self.annotations.append(annotations[key])
               
    @property
    def num_tasks(self):
        return len(self.annotations)
    
    @staticmethod
    def generate_image_grid(image_path_list, grid_size, save_path):
        """
        Generate a grid image from a list of image paths.

        Args:
            image_path_list (list of str): List of image file paths.
            grid_size (tuple): Tuple of (rows, cols), e.g., (10, 10).
            save_path (str): Output path to save the generated grid image.
        """
        rows, cols = grid_size
        assert len(image_path_list) >= rows * cols, "Not enough images to fill the grid"

        # Load the first image to get the size
        sample_img = Image.open(image_path_list[0])
        img_width, img_height = sample_img.size

        # Create a blank canvas
        grid_img = Image.new('RGB', (cols * img_width, rows * img_height))

        # Paste images onto the grid
        for idx, img_path in enumerate(image_path_list[:rows * cols]):
            img = Image.open(img_path).resize((img_width, img_height))
            row, col = divmod(idx, cols)
            grid_img.paste(img, (col * img_width, row * img_height))

        # Save the final grid
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid_img.save(save_path)
        print(f"Saved grid image to {save_path}")

if __name__ == '__main__':
    path_list = [
        'vlm-agibot-4',
        'vlm-bdd10k-2',
        'vlm-epickitchen-4',
        'vlm-rt1-3',
        'vlm-where2place-2'
    ]
    
    for path in path_list:
        task = Task('data/' + path + '/train/')
        task.load_task()
        print(task.num_tasks)
        
        # generate examples grid
        # for i in range(task.num_tasks):
        #     ply, h, w, image_path, lang = task.get_task(i)
        #     image_path = 'data/' + path + '/train/' + image_path
        #     save_path = 'results/' + 'vis/' + path
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     task.visualize_polygon(image_path, ply, save_path + f'/vis_{i}.png', lang)
            
        
        # randomly sample 10 images from 0-task.num_tasks-1
        grid_list = []
        for i in range(16):
            task_id = np.random.randint(0, task.num_tasks)
            grid_list.append('results/' + 'vis/' + path + '/vis_' + str(task_id) + '.png')
        
        # generate grid
        task.generate_image_grid(grid_list, (4, 4), 'results/' + 'vis/' + path + '/grid.png')
        
        
        