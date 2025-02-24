export model_name="<model name>"
export model_path="path/to/model"
export output_dir="./result"
export OPENAI_API_KEY="<Your OpenAI api key here>"

CUDA_VISIBLE_DEVICES=0 python src/gen_model_response.py \
    --model-name ${model_name} \
    --model-path ${model_path} \
    --data-dir ./data \
    --output-dir ${output_dir}


python src/multi_agent_eval.py \
    --model-name ${model_name} \
    --judge-name gpt-4o-mini \
    --output-dir ${output_dir} \
    --parallel 8


python src/show_result.py \
    --model-name ${model_name} \
    --judge-name gpt-4o-mini \
    --output-dir ${output_dir} 