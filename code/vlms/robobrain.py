# insert system path of 
from RoboBrain.inference import SimpleInference
from .base import BaseVLM, ReturnedCoordinate
import json
from .base import BaseVLM, ReturnedCoordinate, Point
from typing import List, Tuple, Optional
import re
# model_id = "BAAI/RoboBrain"
# model = SimpleInference(model_id)

# prompt = "Given the obiects in the image, if you are required to complete the task \"Put the apple in the basket\", what is your detailed plan? Write your plan and explain it in detail, using the following format: Step_1: xxx\nStep_2: xxx\n ...\nStep_n: xxx\n"

# image = "./assets/demo/planning.png"

# pred = model.inference(prompt, image, do_sample=True)
# print(f"Prediction: {pred}")

# ''' 
# Prediction: (as an example)
#     Step_1: Move to the apple. Move towards the apple on the table.
#     Step_2: Pick up the apple. Grab the apple and lift it off the table.
#     Step_3: Move towards the basket. Move the apple towards the basket without dropping it.
#     Step_4: Put the apple in the basket. Place the apple inside the basket, ensuring it is in a stable position.
# '''


class RoboBrain(BaseVLM):
    def __init__(self, model="BAAI/RoboBrain"):
        super().__init__()
        if model is None:
            model = "BAAI/RoboBrain"
        self.model = SimpleInference(model)
        self.question_template = 'prompts/robobrain.txt'

    
    def get_point_type(self):
        return 'bbx'
    

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
    




    def parse_trajectory(self, output_str: str) -> Optional[List[Point]]:
        """
        Parse a string like '[[0.1, 0.2], [0.3, 0.4], ...]'
        into a list of at most the first 10 Point(x, y) objects.
        
        - Handles both strict JSON and truncated text.
        - Ignores incomplete trailing pairs.
        - Returns None if no valid pairs are found.
        """
        _NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        _PAIR_PATTERN = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
        # 1) Try strict JSON first
        try:
            data = json.loads(output_str)
            if isinstance(data, list):
                points = [Point(x=float(p[0]), y=float(p[1]))
                        for p in data if isinstance(p, list) and len(p) == 2]
                return points[:10] if points else None
        except Exception:
            pass  # fall back to regex
        
        # 2) Fallback: regex extraction
        points = []
        for m in _PAIR_PATTERN.finditer(output_str):
            x_str, y_str = m.groups()
            try:
                points.append(Point(x=float(x_str), y=float(y_str)))
                if len(points) == 10:
                    break
            except ValueError:
                continue
        
        return points if points else None



    def s3(self, question, ee_pose=None, image=None, prompt_path='prompts/s3/general_robobrain.txt'):
        question = self.preprocess_question_s3(question, None, prompt_path)
        pred = self.model.inference(question, image, do_sample=True)
        print(pred)
        return self.parse_trajectory(pred)
                
        
    def __call__(self, question, image, system_prompt=None):
        question = self.preprocess_question(question)
        pred = self.model.inference(question, image, do_sample=True)
        ans = self.parse_bbox(pred)
        x1, y1, x2, y2 = ans if ans is not None else (0, 0, 0, 0)
        # clip to 0, 1
        x1 = max(0, min(1, x1))
        x2 = max(0, min(1, x2))
        y1 = max(0, min(1, y1))
        y2 = max(0, min(1, y2))
        
        return x1, x2, y1, y2
       
        
        
        

        




