
import os
from .base import BaseVLM, ReturnedCoordinate
import ast
import os
import PIL.Image as Image
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests



class DSKVL(BaseVLM):
    def __init__(self, model_name='allenai/Molmo-7B-D-0924'):
        self.question_template = 'prompts/molmo.txt'
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        
        
    def get_point_type(self):
        return 'point'
    
    def preprocess_image(self, image_path):
        return image_path
    
    # def preprocess_question(self, question):
        
    #     with open(self.question_template, "r") as question_file:
    #         question_template = question_file.read()
    #     prompt_template = PromptTemplate.from_template(question_template)
    #     prompt_template.invoke({"question": question})
    #     return prompt_template.invoke({"question": question}).text
        
    
    def __call__(self, question, image):
        # process the image and text
        image = self.preprocess_image(image)
        question = self.preprocess_question(question)
        inputs = self.processor.process(
            images=[Image.open(image).raw],
            text=question
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)
        
        