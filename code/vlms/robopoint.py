
import os
from .base import BaseVLM, ReturnedCoordinate
import ast
import os
import PIL.Image as Image
import json
import torch

# insert sys path as ../RoboPoint
# import sys
# sys.path.insert(0, '../RoboPoint')
from robopoint.model.builder import load_pretrained_model
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates, SeparatorStyle


class RoboPoint(BaseVLM):
    def __init__(self, model='wentao-yuan/robopoint-v1-vicuna-v1.5-13b', conv_mode='llava_v1'):
        self.question_template = 'prompts/robopoint.txt'
        
        model_path = model
        model_name = get_model_name_from_path(model_path)
        model_base = None
        # load model
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_mode = conv_mode
        
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
        
    
    def __call__(self, question, image):
        image = self.preprocess_image(image)
        question = self.preprocess_question(question)
        
        # run robopoint eval
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # image token
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(image).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        results = ast.literal_eval(outputs)
        return results
        
        