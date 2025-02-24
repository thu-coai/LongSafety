import json
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def load_model(model_path, tokenizer_path):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('loading model...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device).eval()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    print('finish loading')

    return model, tokenizer, device

def load_data(data_dir):
    with open(os.path.join(data_dir, 'longsafety_meta.jsonl'), 'r') as f:
        meta = [json.loads(line) for line in f]

    with open(os.path.join(data_dir, 'longsafety_docs.jsonl'), 'r') as f:
        docs = [json.loads(line) for line in f]

    data = []
    for m, d in zip(meta, docs):
        assert m['id'] == d['id']
        data.append({**m, **d})

    return data


def build_prompt(d):
    query = d['instruction']
    context = d['context']
    query_front_prompt = f'Based on the following long context, {query}\n\n{context}'
    query_end_prompt = f'{context}\n\nBased on the long context above, {query}'
    return query_front_prompt, query_end_prompt


def generate(prompt, model, tokenizer, device):
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def generate_answers(args):
    model, tokenizer, device = load_model(args.model_path, args.tokenizer_path)
    data = load_data(args.data_dir)
    
    results = []
    for d in tqdm(data):
        query_front_prompt, query_end_prompt = build_prompt(d)
        query_front_response = generate(query_front_prompt, model, tokenizer, device)
        query_end_response = generate(query_end_prompt, model, tokenizer, device)
        d['response_front'] = query_front_response
        d['response_end'] = query_end_response
        results.append(d)

    return results

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="The model name to evaluate.")
    parser.add_argument("--model-path", type=str, required=True, help="The model path to load.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="The tokenizer path to load.")
    parser.add_argument("--data-dir", type=str, default='../data', help="The data directory path.")
    parser.add_argument("--output-dir", type=str, default='../result', help="The output directory path.")
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    results = generate_answers(args)
    
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    output_path = f'{args.output_dir}/{args.model_name}/response.jsonl'

    with open(output_path, 'w') as fw:
        for res in results:
            fw.write(json.dumps(res) + '\n')