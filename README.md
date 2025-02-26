# LongSafety: Evaluating Long-Context Safety of Large Language Models

<div align="center">
<img src="assets/longsafety.png" alt="LongSafety" width="70%" />
</div>

This is the codebase for our paper "[LongSafety: Evaluating Long-Context Safety of Large Language Models](https://arxiv.org/abs/2502.16971)".

LongSafety is the first benchmark to comprehensively evaluate LLM safety in open-ended long-context tasks. It encompasses 1,543 instances with an average length of 5,424 words and comprises 7 safety issues and 6 task types, covering a wide range of long-context safety problems in real-world scenarios.

## News

**🎉 `2025/02/26`:** We have released the data of LongSafety on [Huggingface🤗](https://huggingface.co/datasets/thu-coai/LongSafety).

**🎉 `2025/02/25`:** We have released the data and evaluation code of LongSafety.

## Data

The data of LongSafety is placed in `data/longsafety_meta.jsonl` and `data/longsafety_docs.jsonl`. The meta information of each instance is included in `data/longsafety_meta.jsonl` with the following format:

- `id` (integer): A unique indentifier of the instance.
- `link` (list): The web links of the source documents.
- `length` (integer): The word length of the long context.
- `safety_type` (string): The safety type of the instance.
- `key_words` (list): The safety keywords of the instance.
- `instruction` (string): The safety instruction.
- `task_type` (string): The task type of the instruction.
- `doc_num` (integer): The number of documents integrated in the long context.

The long context of each instance is included in `data/longsafety_docs.jsonl`. The format is as follows:

- `id` (integer): A unique indentifier of the instance.
- `context` (string): The actual long context.

The instances in the two files are associated by the key `id`.

## Evaluation

To run the evaluation codes, please first install the necessary packages.

```bash
pip install -r requirements.txt
```

Then you can evaluate any desired models via `eval.sh`:

```bash
bash eval.sh
```

You can change the `model_name` and `model_path` in `eval.sh` to evaluate different models. You will also need to set your openai api key at `OPENAI_API_KEY` in `eval.sh` to use the multi-agent framework for safety judgment of the generation results. The results will be saved in `./result` by default, and you can freely change it if necessary.

## Citation

Please kindly cite our paper if you find our work helpful.

```
@misc{lu2025longsafetyevaluatinglongcontextsafety,
      title={LongSafety: Evaluating Long-Context Safety of Large Language Models}, 
      author={Yida Lu and Jiale Cheng and Zhexin Zhang and Shiyao Cui and Cunxiang Wang and Xiaotao Gu and Yuxiao Dong and Jie Tang and Hongning Wang and Minlie Huang},
      year={2025},
      eprint={2502.16971},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.16971}, 
}
```
