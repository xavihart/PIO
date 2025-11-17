
import os
from .base import BaseVLM, ReturnedCoordinate, Point
import ast
import os
import PIL.Image as Image
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

import re
from typing import List, Tuple

def parse_points_from_string(s: str) -> List[Tuple[float, float]]:
    # Match attributes starting with x or x<number>
    x_vals = re.findall(r'\bx\d*="([\d.]+)"', s)
    y_vals = re.findall(r'\by\d*="([\d.]+)"', s)

    if len(x_vals) != len(y_vals):
        raise ValueError("Mismatched number of x and y values")

    return [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]


class Molmo(BaseVLM):
    def __init__(self, model='allenai/Molmo-7B-D-0924'):
        self.question_template = 'prompts/molmo.txt'
        self.model_name = model
        self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto',
                cache_dir='/ssdscratch/hxue45/.cache/'
            )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            cache_dir='/ssdscratch/hxue45/.cache/'
        )
        
        
        
    def get_point_type(self):
        return 'points'
    
    def preprocess_image(self, image_path):
        return image_path
    
    # def preprocess_question(self, question):
        
    #     with open(self.question_template, "r") as question_file:
    #         question_template = question_file.read()
    #     prompt_template = PromptTemplate.from_template(question_template)
    #     prompt_template.invoke({"question": question})
    #     return prompt_template.invoke({"question": question}).text
    def s3(self, question, ee_pose, image, prompt_path='prompts/s3/general_molmo.txt'):
        base64_image = self.preprocess_image(image)
        
        # for molmo since the point coordinate is from 0-100 normalization is needed
        
        question = self.preprocess_question_s3(question, ee_pose, prompt_path)
        inputs = self.processor.process(
            images=[Image.open(image)],
            text=question
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", do_sample=True),
            tokenizer=self.processor.tokenizer
        )
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        points = parse_points_from_string(generated_text)
        # normalize it by /100
        points = [(x/100, y/100) for x, y in points]
        if len(points) == 0:
            # print(question)
            print(generated_text)
            raise ValueError("No points found")
        
        output = []
        for point in points:
            output.append(Point(x=point[0], y=point[1]))
        return output
        
    def __call__(self, question, image):
        # process the image and text
        image = self.preprocess_image(image)
        question = self.preprocess_question(question)
        
        inputs = self.processor.process(
            images=[Image.open(image)],
            text=question
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", do_sample=True),
            tokenizer=self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # extract points from the generated text
        print(generated_text)
        points = parse_points_from_string(generated_text)
        
        # normalize it by /100
        points = [(x/100, y/100) for x, y in points]
        
        if len(points) == 0:
            raise ValueError("No points found")
        
        return points
        
        