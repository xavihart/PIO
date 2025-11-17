
import os
from .base import BaseVLM, ReturnedCoordinate, Point
import ast
import os
import PIL.Image as Image
import json
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO

import re
from typing import List, Tuple

CACHE_DIR = '/ssdscratch/hxue45/.cache/'

def parse_points_from_string(s: str) -> List[Tuple[float, float]]:
    # Match attributes starting with x or x<number>
    x_vals = re.findall(r'\bx\d*="([\d.]+)"', s)
    y_vals = re.findall(r'\by\d*="([\d.]+)"', s)

    if len(x_vals) != len(y_vals):
        raise ValueError("Mismatched number of x and y values")

    return [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]


class MolmoAct(BaseVLM):
    def __init__(self, model='allenai/MolmoAct-7B-D-0812'):
        self.question_template = 'prompts/molmoact.txt'
        self.model_name = model
        self.processor = AutoProcessor.from_pretrained(
                            model,
                            trust_remote_code=True,
                            torch_dtype="bfloat16",
                            device_map="auto",
                            padding_side="left",
                            cache_dir=CACHE_DIR
                        )

        # load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
                        model,
                        trust_remote_code=True,
                        torch_dtype="bfloat16",
                        device_map="auto",
                        cache_dir=CACHE_DIR
                    )
        
        # move to gpu
        self.model.to('cuda')

        
        
        
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
    def s3(self, question, ee_pose, image, prompt_path='prompts/s3/general_molmoact.txt'):
        
        # for molmo since the point coordinate is from 0-100 normalization is needed
        
        question = self.preprocess_question_s3(question, ee_pose, prompt_path)


        text = self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [dict(type="text", text=question)]
                    }
                ], 
                tokenize=False, 
                add_generation_prompt=True,
            )
    
        inputs = self.processor(
                    images=[Image.open(image)],
                    text=text,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
    
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        trace = self.model.parse_trace(generated_text)
        print(f"generated visual reasoning trace: {trace}")
        if eval(str(trace)) == []:
            return [Point(x=0, y=0), Point(x=0, y=0)]

        # parse
        # h, w = inputs['images'].shape[2], inputs['images'].shape[3]
        points = [Point(x=p[0] / 255, y=p[1] / 255) for p in eval(str(trace))[0]]
        return points




        # inputs = self.processor.process(
        #     images=[Image.open(image)],
        #     text=question
        # )
        # inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        # output = self.model.generate_from_batch(
        #     inputs,
        #     GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>", do_sample=True),
        #     tokenizer=self.processor.tokenizer
        # )
        # generated_tokens = output[0,inputs['input_ids'].size(1):]
        # generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # points = parse_points_from_string(generated_text)
        # # normalize it by /100
        # points = [(x/100, y/100) for x, y in points]
        # if len(points) == 0:
        #     # print(question)
        #     print(generated_text)
        #     raise ValueError("No points found")
        
        # output = []
        # for point in points:
        #     output.append(Point(x=point[0], y=point[1]))
        # return output
        
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
        
        