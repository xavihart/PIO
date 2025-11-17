python code/test_s1s2.py --test_models claude-3.7 --open_coord --skip_runned
python code/test_s1s2.py --test_models gpt-4o --open_coord --skip_runned
python code/test_s1s2.py --test_models gpt-4o-mini --open_coord --skip_runned
python code/test_s1s2.py --test_models gpt-o4-mini --open_coord --skip_runned
python code/test_s1s2.py --test_models gpt-o3 --open_coord --skip_runned
python code/test_s1s2.py --test_models gemini-2.0-flash --skip_runned
python code/test_s1s2.py --test_models gemini-2.5-flash --skip_runned
python code/test_s1s2.py --test_models gemini-2.5-pro --skip_runned

python code/test_s1s2.py --test_model robopoint
python code/test_s1s2.py --test_model qwen2.5-vl
python code/test_s1s2.py --test_model molmo-7b
python code/test_s1s2.py --test_model molmo-72b
python code/test_s1s2.py --test_model robobrain


python code/test_s1s2.py --test_model gpt-4o-point --max_cases 4
python code/test_s1s2.py --test_model gemini-2.5-flash-point --max_cases 4

# test_s3
python code/test_s3.py --test_model gpt-4o;
python code/test_s3.py --test_model gpt-o3;
python code/test_s3.py --test_model gemini-2.5-flash;
python code/test_s3.py --test_model gemini-2.5-pro;
python code/test_s3.py --test_model molmo-7b;
python code/test_s3.py --test_model robobrain;
python code/test_s3.py --test_model molmoact;


# run roborefer api
python api.py \
--port 12123 \
--depth_model_path /ssdscratch/hxue45/model_weights/depth_anything_v2.pth \
--vlm_model_path /ssdscratch/hxue45/model_weights/RoboRefer-8B-SFT


# cal score


