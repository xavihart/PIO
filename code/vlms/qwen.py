import base64
from openai import OpenAI 
import os
from .base import BaseVLM, ReturnedCoordinate, Point
import ast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



class QWen_VL(BaseVLM):
    def __init__(self, model="Qwen/Qwen2.5-VL-32B-Instruct"):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model, torch_dtype="auto", device_map="auto", cache_dir='/ssdscratch/hxue45/.cache/',
                # attn_implementation="flash_attention_2"
            )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.question_template = 'prompts/qwen.txt'

    
    def get_point_type(self):
        return 'bbx'
    
    def preprocess_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    @staticmethod
    def parse_json(json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output
    
    def s3(self, question, ee_pose, image, prompt_path='prompts/s3/general_qwen.txt'):
        base64_image = self.preprocess_image(image)
        question = self.preprocess_question_s3(question, ee_pose, prompt_path)
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
        text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1028)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        
        traj = output_text[0]
        import re
        import json
        print(traj)
        cleaned = traj.strip('`json\n')

        # Step 2: Remove C-style comments
        cleaned = re.sub(r'//.*', '', cleaned)

        # Step 3: Replace (x, y) with [x, y]
        cleaned = re.sub(r'\(([^)]+)\)', r'[\1]', cleaned)

        # Step 4: Parse JSON
        
        parsed_dict = json.loads(cleaned)
        points = parsed_dict['trajectory']
        
        output = []
        for point in points:
            output.append(Point(x=point[0], y=point[1]))
        return output
                
        
    def __call__(self, question, image, system_prompt=None):
        base64_image = self.preprocess_image(image)
        question = self.preprocess_question(question)
        
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
        text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        
        bbx = output_text[0]
        print(bbx)
        json_output = ast.literal_eval(self.parse_json(bbx))
        # normalize
        x1,y1,x2,y2 = json_output[0]['bbox_2d']
        x1 = x1/input_width.item()
        x2 = x2/input_width.item()
        y1 = y1/input_height.item()
        y2 = y2/input_height.item()
        # crop to be smaller than 1
        x1 = min(x1, .99)
        y1 = min(y1, .99)
        x2 = min(x2, .99)
        y2 = min(y2, .99)
        return (x1, y1, x2, y2)
       
        
        
        

        




