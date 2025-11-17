from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from typing import List, Tuple
import cv2
from matplotlib import pyplot as plt
import numpy as np

class ReturnedCoordinate(BaseModel):
    explanations: str
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    
    
class ReturnedCoordinateFloat(BaseModel):
    explanations: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

class Point(BaseModel):
    x: float
    y: float

class Point_Int(BaseModel):
    x: int
    y: int

class Point_Float(BaseModel):
    x: float
    y: float


class ReturnedTrajectory(BaseModel):
    explanations: str
    trajectory: List[Point_Float]
    
    
class Points_Int(BaseModel):
    explanations: str
    points: List[Point_Int]
    
class Points_Float(BaseModel):
    explanations: str
    points: List[Point_Float]
    
class BaseVLM(object):
    def __init__(self, ):
        self.question_template = 'prompts/general_bbx.txt'
        pass
    
    def preprocess_image(self, image_path):
        return image_path
    
    @staticmethod
    
    def draw_xy_axis_numbers(image_path, save_path):
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Set DPI and compute adaptive figure size
        dpi = 100
        figsize = (width / dpi, height / dpi)

        # Generate 8 tick intervals
        x_ticks = np.linspace(0, width - 1, 8)
        y_ticks = np.linspace(0, height - 1, 8)

        # Plot with adaptive size and fixed resolution
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image_rgb)
        plt.xticks(x_ticks, labels=[f'{int(x)}' for x in x_ticks], fontsize=8)
        plt.yticks(y_ticks, labels=[f'{int(y)}' for y in y_ticks], fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
    
    def preprocess_question(self, question, prompt_path=None):
        if prompt_path is None:
            prompt_path = self.question_template
        with open(prompt_path, "r") as question_file:
            question_template = question_file.read()
        prompt_template = PromptTemplate.from_template(question_template)
        prompt_template.invoke({"question": question})
        return prompt_template.invoke({"question": question}).text
    
    def preprocess_question_s3(self, task, ee_pose, prompt_path=None):
        assert prompt_path is not None, "prompt_path should not be None"
        with open(prompt_path, "r") as question_file:
            question_template = question_file.read()
        prompt_template = PromptTemplate.from_template(question_template)
        q = prompt_template.invoke({"task": task, "eepose": ee_pose}).text
        return q
        
    def s3(self, question, ee_pose, image):
        # call function for s3 inference
        pass
    
    def __call__(self, question, image):
        # call function for s1 s2 inference
        # return (min_x, min_y, max_x, max_y)
        # should be scaled to original image size
        pass