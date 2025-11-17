import argparse
import re
import json
import cv2
import numpy as np
from RoboRefer.API.query_model import query_server
from PIL import Image
import cv2
from .base import BaseVLM

def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg",
                         color=(244, 133, 66), radius=12, border_color=(255, 255, 255), border_thickness=2):
    """
    Denormalizes normalized points and marks them on the image with a colored circle and white border.
    
    Args:
        image_path (str): Path to the input image.
        normalized_points (list of tuple): List of (x, y) in normalized coordinates [0, 1].
        output_path (str): Where to save the annotated image.
        color (tuple): BGR color of the inner circle.
        radius (int): Radius of the inner circle.
        border_color (tuple): BGR color of the circle's white border.
        border_thickness (int): Thickness of the border around the circle.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    for nx, ny in normalized_points:
        x = int(nx * width)
        y = int(ny * height)
        # Draw outer white border
        cv2.circle(image, (x, y), radius + border_thickness, border_color, thickness=-1)
        # Draw inner colored circle
        cv2.circle(image, (x, y), radius, color, thickness=-1)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")

class RoboReferAPI(BaseVLM):
    def __init__(self, model=None):
        super().__init__()
        pass
        # calling the runned API
        # python use_api.py --image_path ../assets/tabletop.jpg --prompt "Pick the apple in front of the logo side of the leftmost cup." --output_path ../assets/my_tabletop_result_1.jpg --url http://127.0.0.1:12123
    


    def model_inference(self, question, image_path, url="http://127.0.0.1:12123"):
        suffix = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)],  where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."

        test_image_paths = [image_path]

        answer = query_server(
            test_image_paths,
            question + suffix,
            url=url,
            enable_depth=1
        )

        normalized_points = eval(answer.strip())
        print(f"Normalized points: {normalized_points}")

        return normalized_points



    def get_point_type(self):
        return 'points'
    

    def parse_bbox(self, output_str):
        """
        Parse x, y, w, h from a JSON-like list string.
        Returns (x, y, w, h) if successful, else None.
        """
        try:
            data = json.loads(output_str)
            if isinstance(data, list) and len(data) == 4:
                x, y, w, h = data
                return x, y, w, h
            return None
        except Exception:
            return None
    



    def s3(self, question, ee_pose=None, image=None, prompt_path='prompts/s3/general_robobrain.txt'):
        question = self.preprocess_question_s3(question, None, prompt_path)
        pred = self.model.inference(question, image, do_sample=True)
        print(pred)
        return self.parse_trajectory(pred)
                
        
    def __call__(self, question, image, system_prompt=None):
        question = 'The task is to locate the point referred in the following questions: {question}.\n'
        ans = self.model_inference(question, image)
        return ans
       
        
        
        

        




