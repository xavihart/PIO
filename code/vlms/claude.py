import base64
import os
from .base import BaseVLM, ReturnedCoordinate
import ast
import os
import PIL.Image
import json
import anthropic
import base64
import cv2

class Claude(BaseVLM):
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        super().__init__()
        self.client = anthropic.Anthropic()
        self.model = model
        # self.question_template = 'prompts/general_bbx_old.txt'
    
    def get_point_type(self):
        return 'bbx'
    
    def preprocess_image(self, image_path, open_coord=False):
        # draw the image with coordinates and save it
        
        if not open_coord:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            # draw the image with coordinates and save it
            if os.path.exists('.tmp.jpg'):
                os.remove('.tmp.jpg')
            save_path = '.tmp.jpg'
            self.draw_xy_axis_numbers(image_path, save_path)
            with open(save_path, "rb") as image_file:
                base_ =  base64.b64encode(image_file.read()).decode("utf-8")
            # remove the tmp file
            os.remove(save_path)
            return base_

    # def preprocess_question(self, question):
    #     # load question template
    #     with open(self.question_template, "r") as question_file:
    #         question_template = question_file.read()
    #     question = question_template + '\n' + question
    #     return question
    
    def __call__(self, question, image, system_prompt=None, open_coord=False):
        
        image_base = self.preprocess_image(image, open_coord=open_coord)
        question = self.preprocess_question(question)
        
        text_analysis_schema = ReturnedCoordinate.model_json_schema()
        tools = [
                {
                    "name": "ReturnedCoordinate",
                    "description": "ReturnedCoordinate (min_x, min_y, max_x, max_y)",
                    "input_schema": text_analysis_schema
                }
            ]
        message = self.client.messages.create(
                            model=self.model,
                            max_tokens=1024,
    
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": image_base,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": question
                                        }
                                    ],
                                }
                            ],
                            tools=tools,
                            tool_choice={"type": "tool", "name": "ReturnedCoordinate"}
                        )
        function_call = message.content[0].input
        coord = ReturnedCoordinate(**function_call)
        
    
        min_x = coord.min_x
        min_y = coord.min_y
        max_x = coord.max_x
        max_y = coord.max_y
    
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




