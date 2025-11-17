# Point-It-Out (PIO): Benchmarking Embodied Reasoning for Vision-Language Models in Multi-Stage Visual Grounding

> TLDR: A unified benchmark for embodied reasoning in realistic embodied scenes in format of visual grounding.

<p align="center">
  <img src="src/logo.png" alt="PIO Benchmark Overview" width="80%">
</p>

**Authors**  
Haotian Xue¹²*, Yunhao Ge², Yu Zeng², Zhaoshuo Li², Ming-Yu Liu², Yongxin Chen¹², Jiaojiao Fan²  
¹ Georgia Tech  ² NVIDIA  

---

## Introduction
Different from other embodied reasoning benchmarks that always rely on Multi-Choice QA (MCQ) to choose right anwers (e.g. ABCD), we switch to visual grounding to test whether the Vision-Language Model really knows where to point to in embodied scenarios. We believe it is an important way to really connect multi-modal language models to the physical-world. 


## Three-Stage Design
We design a three-stage evaluation in embodied scenarios, with 
- (S1): Refer specific object with different constraints e.g. location, object part and appearance;
- (S2) Refer to somewhere based on specific task e.g. recommendation, affordance, next-state prediction;
- (S3) Visual trace prediction, which requires the model to predict the visual trace to complete certain task.


## PIO Benchmarks

It contains around 600 questions (230 for S1, 270 for S2 and 100 for S3)
 - for S1 and S2 we have human-annotated ground-truth in format of segmentation masks; 
- for S3 we provide a VLM prompt to judge the quality of generated visual traces.

Here is some useful code about how to load the dataset of S1 and S2
```python
import json
s1s2_data=json.load(open('data/s1s2.json', 'rb'))
for sample in range(len(s1s2_data)):
    polygon = sample['polygon'] # ground-truth seg mask
    image_path = sample['image_path']
    height, width = sample['height'], sample['width']
    prompt = sample['lang'] # task prompt
    s1_or_s2 = sample['s1_or_s2'] # s1 or s2
    subclass = sample['subclasses'] # s1s2 subclasses e.g. recommendation
```
the  root of `image_path` is `data/images_s1s2/`;

For S3 quesionts we can load in similar way:


```python
import json
s3_data=json.load(open('data/s3.json', 'rb'))
for sample in range(len(s3_data)):
    image_path = sample['image_path']
    prompt = sample['lang']
```
the  root of `image_path` is `data/images_s3/`;



## Demo Inference on Gemini-2.5-pro

Here we use Gemini-2.5-pro as an example to show you how to do inference on the whole benchmark by calling   `code/test_s1s2.py` for S1/S2 and  `code/test_s3.py` for S3.

The following script can be used to test running inference on s1/s2:
```bash
EXP_NAME="demo"
MAX_CASES=5 # set to -1 if run on all cases
python code/test_s1s2.py \
  --test_models gemini-2.5-flash \
  --max_cases ${MAX_CASES} \
  --exp_name ${EXP_NAME} \
  --which_s both \
  --save_path results \
  --s1s2_path data/s1s2.json
```
for s3 you can test with the following script:
```bash
EXP_NAME="demo"
MAX_CASES=5 # set to -1 if run on all cases
python code/test_s3.py \
  --test_model gemini-2.5-flash \
  --max_cases ${MAX_CASES} \
  --exp_name ${EXP_NAME} \
  --json_path data/s3.json \
  --image_root data/images_s3 \
  --save_path results

```
after running, you can see results saved in results/, with visualized answer and ground-truth `vis.png`; and also prediction results in `info.npy`; the visualizaion is like (the 3 rows are for S1/S2/S3 in order)

<p>
  <img src="results/s1/demo_gemini_s1s2_111705/3/results.png" alt="PIO Benchmark Overview" width="30%">
 <img src="results/s1/demo_gemini_s1s2_111705/5/results.png" alt="PIO Benchmark Overview" width="30%">
 <img src="results/s1/demo_gemini_s1s2_111705/6/results.png" alt="PIO Benchmark Overview" width="30%">
</p>
<p>
  <img src="results/s2/demo_gemini_s1s2_111705/4/results.png" alt="PIO Benchmark Overview" width="30%">
 <img src="results/s2/demo_gemini_s1s2_111705/7/results.png" alt="PIO Benchmark Overview" width="30%">
 <img src="results/s2/demo_gemini_s1s2_111705/2/results.png" alt="PIO Benchmark Overview" width="30%">
</p>

<p>
  <img src="results/s3/demo_gemini_s3_111705/1/vis.png" alt="PIO Benchmark Overview" width="30%">
  <img src="results/s3/demo_gemini_s3_111705/2/vis.png" alt="PIO Benchmark Overview" width="30%">
  <img src="results/s3/demo_gemini_s3_111705/3/vis.png" alt="PIO Benchmark Overview" width="30%">
</p>

## Test New Models
The VLM zoo is put in file `code/vlms/__init__.py`, where you can define your own VLMs inside by 
- put your model_name and model_class inside `get_vlm()` function; 
- implement the model class as `new_model.py` and put it under `code/vlms/`, including some important function like `__call__()`, `preprocess_image()` and `s3()`; `__call___()` is by default used for S1/S2 and `s3()` serves as the call fucntion to run S3 inference
- Finally, you need to write the prompt for your new model, and put the path of prompt into `self.question_template`;

We provide some demo code for models including GPT-series, Gemini-series, MoLMO, RoboRefer and RoboBrain inside, and the propmts are in `prompts/`


## Evaluation
For evaluation of S3, we provide the code to evaluate S3 in  `prompts/s3_auto.txt`; we suggest use human annotators to evaluate it since it is more reliable.

For evaluation of S1 or S2, we provide an evaluation code for evaluation and plot drawing in    `code/eval_s1s2.py`.