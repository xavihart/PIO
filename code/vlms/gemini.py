import base64
from openai import OpenAI 
import os
from .base import BaseVLM, ReturnedCoordinate, ReturnedTrajectory, ReturnedCoordinateFloat, Points_Float
import ast
from google import genai
from google.genai import types
import os
import PIL.Image
import json
import cv2



class Gemini(BaseVLM):
    def __init__(self, model="gemini-2.0-flash"):
        super().__init__()
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        self.model = model
        self.question_template = 'prompts/general_bbx_old.txt'
        if 'point' in model:
            self.model = model.replace('-point', '')
            self.return_points = True
            self.point_type = 'points'
            self.prompt_template = 'prompts/general_points.txt'
        else:
            self.point_type = 'bbx'
    
    def get_point_type(self):
        return self.point_type
    
    def preprocess_image(self, image_path, open_coord=False):
        if not open_coord:
            return PIL.Image.open(image_path)
        else:
            if os.path.exists('.tmp.png'):
                os.remove('.tmp.png')
            save_path = '.tmp.png'
            self.draw_xy_axis_numbers(image_path, save_path)
            pil_image =  PIL.Image.open(save_path)
            # remove the tmp file
            os.remove(save_path)
            return pil_image
        
    
    
    # def preprocess_question(self, question):
    #     # load question template
    #     with open(self.question_template, "r") as question_file:
    #         question_template = question_file.read()
    #     question = question_template + '\n' + question
    #     return question
    
    def s3(self, question, ee_pose, image, prompt_path='prompts/s3/general.txt'):
        image = self.preprocess_image(image)
        question = self.preprocess_question_s3(question, ee_pose, prompt_path)
        response = self.client.models.generate_content(
                    model=self.model,
                    contents=[question, image],
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': ReturnedTrajectory,
                    },)
        output = response.parsed
        return output.trajectory
    
    def __call__(self, question, image, system_prompt=None, open_coord=False):  
        image_pil = self.preprocess_image(image, open_coord=open_coord)
        question = self.preprocess_question(question)
        response = self.client.models.generate_content(
                    model=self.model,
                    contents=[question, image_pil],
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': ReturnedCoordinateFloat if self.point_type == 'bbx' else Points_Float,
    },)
        results = response.parsed
        
        if self.point_type == 'points':
            points = results.points
            points = [(point.x, point.y) for point in points]
            # crop
            points = [(min(x,1), min(y,1)) for x, y in points]
            return points
            
        else:
            min_x, min_y, max_x, max_y = results.min_x, results.min_y, results.max_x, results.max_y
            # normalize
            print(min_x, min_y, max_x, max_y)
            # image_data = cv2.imread(image)
            # height, width, _ = image_data.shape
            # min_x = min_x / width
            # min_y = min_y / height
            # max_x = max_x / width
            # max_y = max_y / height
            
            # crop to 0, 1
            min_x = min(min_x, 1)
            min_y = min(min_y, 1)
            max_x = min(max_x, 1)
            max_y = min(max_y, 1)
            
            return (min_x, min_y, max_x, max_y)




