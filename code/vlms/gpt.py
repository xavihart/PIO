import base64
from openai import OpenAI 
import os
from .base import BaseVLM, ReturnedCoordinate, ReturnedTrajectory, Points_Int
import ast
import cv2


class GPT(BaseVLM):
    def __init__(self, model="gpt-4o"):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
        self.model = model
        self.point_type = 'bbx'
        
        if 'point' in model:
            self.model = model.replace('-point', '')
            self.return_points = True
            self.point_type = 'points'
            self.prompt_template = 'prompts/general_points_withcoord.txt'
    
    def get_point_type(self):
        return self.point_type
    
    def preprocess_image(self, image_path, open_coord=False):
        # draw the image with coordinates and save it
        if not open_coord:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            # draw the image with coordinates and save it
            if os.path.exists('.tmp.png'):
                os.remove('.tmp.png')
            save_path = '.tmp.png'
            self.draw_xy_axis_numbers(image_path, save_path)
            with open(save_path, "rb") as image_file:
                base_ =  base64.b64encode(image_file.read()).decode("utf-8")
            # remove the tmp file
            os.remove(save_path)
            return base_
        
        
    def s3(self, question, ee_pose, image, prompt_path='prompts/s3/general.txt'):
        base64_image = self.preprocess_image(image)
        question = self.preprocess_question_s3(question, ee_pose, prompt_path)
        kwargs = {}
        if 'o3' in self.model or 'o4' in self.model:
            kwargs['reasoning_effort']="high"
            # kwargs['temperature']=1
            kwargs['max_completion_tokens']=100000
        else:
            kwargs['temperature']=0.0
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": f"{question}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            response_format=ReturnedTrajectory,
            **kwargs
        )
        output = response.choices[0].message.parsed
        return output.trajectory
    
    
    def ic_call(self, question, image, example_image='prompts/example_image.png', prompt_path='prompts/ic_general.txt'):
        base64_image = self.preprocess_image(image)
        base64_example_image = self.preprocess_image(example_image)
        
        question = self.preprocess_question('', prompt_path='prompts/ic_general.txt')
        
        kwargs = {}
        
        if 'o3' in self.model or 'o4' in self.model:
            kwargs['reasoning_effort']="high"
            # kwargs['temperature']=1
            kwargs['max_completion_tokens']=100000
        else:
            kwargs['temperature']=0.0
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": f"{question}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_example_image}"}
                    },
                    {"type": "text", "text": f"Now given the new task:\n{question}\n Please give your answer:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            response_format=ReturnedCoordinate if self.point_type == 'bbx' else Points_Int,
            **kwargs
        )
        output = response.choices[0].message.parsed
        min_x, min_y, max_x, max_y = output.min_x, output.min_y, output.max_x, output.max_y
        
        return (min_x, min_y, max_x, max_y)
    
    def __call__(self, question, image, system_prompt=None, open_coord=False):
        base64_image = self.preprocess_image(image, open_coord=open_coord)
        question = self.preprocess_question(question)
        
        kwargs = {}
        
        if 'o3' in self.model or 'o4' in self.model:
            kwargs['reasoning_effort']="high"
            # kwargs['temperature']=1
            kwargs['max_completion_tokens']=100000
        else:
            kwargs['temperature']=0.0
            
            
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": f"{question}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            response_format=ReturnedCoordinate if self.point_type == 'bbx' else Points_Int,
            **kwargs
        )
        # breakpoint()
        
        
        output = response.choices[0].message.parsed
        
        if not self.point_type == 'points':
            min_x, min_y, max_x, max_y = output.min_x, output.min_y, output.max_x, output.max_y

            image_data = cv2.imread(image)
            height, width, _ = image_data.shape
            min_x = min_x / width
            min_y = min_y / height
            max_x = max_x / width
            max_y = max_y / height
            
            # crop to 0, 1
            min_x = min(min_x, 1)
            min_y = min(min_y, 1)
            max_x = min(max_x, 1)
            max_y = min(max_y, 1)
            return (min_x, min_y, max_x, max_y)
        else:
            points = output.points
            image_data = cv2.imread(image)
            height, width, _ = image_data.shape
            normalized_points = [(point.x / width, point.y / height) for point in points]
            # crop to 0, 1
            normalized_points = [(min(x, 1), min(y, 1)) for x, y in normalized_points]
            return normalized_points
                
            




