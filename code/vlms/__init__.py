from .gpt import GPT
from .gemini import Gemini
from .claude import Claude
from .qwen import QWen_VL
from .robopoint import RoboPoint
from .molmo import Molmo
from .deepseek import DSKVL
from .robobrain import RoboBrain
from .roborefer import RoboReferAPI
from .molmo_act import MolmoAct

def get_vlm(vlm_name):
    if vlm_name == 'gpt-4o':
        return GPT, 'gpt-4o'
    if vlm_name == 'gpt-4o-point':
        return GPT, 'gpt-4o-point'
    if vlm_name == 'gpt-4o-mini':
        return GPT, 'gpt-4o-mini'
    if vlm_name == 'gpt-o3':
        return GPT, 'o3'
    if vlm_name == 'gpt-o3-point':
        return GPT, 'gpt-o3-point'
    if vlm_name == 'gpt-o4-mini':
        return GPT, 'o4-mini'
    elif vlm_name == 'gemini-2.0-flash':
        return Gemini, 'gemini-2.0-flash'
    elif vlm_name == 'gemini-2.0-flash-point':
        return Gemini, 'gemini-2.0-flash-point'
    elif vlm_name == 'gemini-2.5-flash':
        return Gemini, 'gemini-2.5-flash-preview-04-17'
    elif vlm_name == 'gemini-2.5-flash-point':
        return Gemini, 'gemini-2.5-flash-preview-04-17-point'
    elif vlm_name == 'gemini-2.5-pro-point':
        return Gemini, 'gemini-2.5-pro-preview-03-25-point'
    elif vlm_name == 'gemini-2.5-pro':
        return Gemini, 'gemini-2.5-pro-preview-03-25'
    elif vlm_name == 'claude-3.7':
        return Claude, 'claude-3-7-sonnet-20250219'
    elif vlm_name == 'qwen2.5-vl':
        return QWen_VL, 'Qwen/Qwen2.5-VL-32B-Instruct'
    elif vlm_name == 'robopoint':
        return RoboPoint, 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'
    elif vlm_name == 'molmo-7b':
        return Molmo, 'allenai/Molmo-7B-D-0924'
    elif vlm_name == 'molmo-72b':
        return Molmo, 'allenai/Molmo-72B-0924'
    elif vlm_name == 'deepseekvl':
        return DSKVL, None
    elif vlm_name == 'robobrain':
        return RoboBrain, 'BAAI/RoboBrain'
    elif vlm_name == 'roborefer':
        return RoboReferAPI, None
    elif vlm_name == 'molmoact':
        return MolmoAct, 'allenai/MolmoAct-7B-D-0812'
    else:
        raise ValueError(f"VLM {vlm_name} not found")