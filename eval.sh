export model_name="Llama-3.1-8B-Instruct"
export output_dir="./result"

CUDA_VISIBLE_DEVICES=0 python src/gen_model_response.py \
    --model-name ${model_name} \
    --model-path /data/public_checkpoints/huggingface_models/Llama-3.1-8B-Instruct \
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