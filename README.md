# Multi-Reference KL-Regularized RLHF

Code for the NeurIPS 2025 paper **"KL-Regularized RLHF with Multiple Reference Models: Exact Solutions and Sample Complexity"**
by Gholamali Aminian, Amir R. Asadi, Idan Shenfeld, and Youssef Mroueh.

[[NeurIPS 2025]](https://neurips.cc/virtual/2025/poster/116496) · [[OpenReview]](https://openreview.net/forum?id=j8XnFfTvXF) · [[arXiv:2502.01203]](https://arxiv.org/abs/2502.01203)

Standard RLHF regularizes the policy towards a single reference model. This repository implements alignment with
**multiple reference models** using the exact solutions derived in the paper:

| Regularizer | Optimal multi-reference policy | `ensemble_type` |
|---|---|---|
| Reverse KL (RKL) | weighted **geometric** mean of the references, $\pi_{\text{ref}} \propto \pi_1^{\alpha}\,\pi_2^{1-\alpha}$ | `geometric` |
| Forward KL (FKL) | weighted **arithmetic** mean of the references, $\pi_{\text{ref}} = \alpha\,\pi_1 + (1-\alpha)\,\pi_2$ | `arithmetic` |

The combined reference is implemented as a drop-in model ([`multiref/ensemble_model.py`](multiref/ensemble_model.py))
and used as the reference policy in both online (GRPO) and offline (DPO) training.

## Repository layout

```
multiref/
  ensemble_model.py      # EnsembleModel: geometric / arithmetic mixture of two causal LMs
  grpo_config.py         # GRPOConfig (TRL) + multi-reference options
  grpo_trainer.py        # GRPOTrainer (TRL) accepting an external (ensemble) reference model
  math_eval/             # math answer extraction and scoring utilities
scripts/
  train_grpo_gsm8k.py    # online RL: GRPO on GSM8K
  train_dpo.py           # offline: DPO on UltraFeedback
  eval_gsm8k_vllm.py     # GSM8K test-set rewards of a GRPO checkpoint (vLLM)
  eval_reward.py         # reward-model scores on the UltraFeedback test split
  reward_distribution.py # reward distribution of sampled responses (single model or ensemble)
```

## Installation

Python 3.10+ and CUDA GPUs are required.

```bash
git clone https://github.com/GholamaliAminian/Multi_reference_Alignment.git
cd Multi_reference_Alignment

conda create -n multiref python=3.10 -y
conda activate multiref

pip install -r requirements.txt
pip install -e .

# optional
pip install vllm        # GSM8K evaluation and vLLM generation during GRPO
pip install flash-attn  # used by scripts/eval_reward.py
```

All scripts are run from the repository root.

## Usage

By default the training scripts use **two GPUs**: the policy on `cuda:0` and the reference model(s) on `cuda:1`.

### Online RL: GRPO on GSM8K

The ensemble reference combines the initial policy (`--model_name`) with a second model (`--ref_model_name`);
`--alpha` is the weight of the first.

```bash
# Reverse KL: geometric mean of the two references
python scripts/train_grpo_gsm8k.py \
  --multi_ref \
  --ensemble_type geometric \
  --alpha 0.5 \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --ref_model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --output_dir outputs/grpo_gsm8k_rkl_a0.5

# Forward KL: arithmetic mean of the two references
python scripts/train_grpo_gsm8k.py --multi_ref --ensemble_type arithmetic --alpha 0.5 \
  --output_dir outputs/grpo_gsm8k_fkl_a0.5

# Single-reference baseline (reference = --ref_model_name)
python scripts/train_grpo_gsm8k.py --output_dir outputs/grpo_gsm8k_single
```

Other options: `--num_generations`, `--num_iterations`, `--max_completion_length`, `--train_batch_size`,
`--gradient_accumulation_steps`, `--policy_device`, `--ref_device`, `--report_to` (default `wandb`).
Run with `--help` for the full list.

To generate with vLLM, start a server on a separate GPU and pass `--use_vllm`:

```bash
CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/train_grpo_gsm8k.py --multi_ref --use_vllm
```

### Offline: DPO on UltraFeedback

`scripts/train_dpo.py` accepts all TRL `DPOConfig` / `ModelConfig` arguments (e.g. `--beta`, `--learning_rate`).
The reference model is selected with `--lora_alpha`:

| `--lora_alpha` | Reference model |
|---|---|
| `0` (default) | single reference: `--model_name_or_path` |
| `a > 0` | geometric mean (RKL), weight `a` on `--model_name_or_path`, `1 - a` on `--ref_model_2` |
| `-a < 0` | arithmetic mean (FKL), weight `a` on `--model_name_or_path`, `1 - a` on `--ref_model_2` |

```bash
# Reverse KL, alpha = 0.5
python scripts/train_dpo.py \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --ref_model_2 Qwen/Qwen2-1.5B-Instruct \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --lora_alpha 0.5 \
  --output_dir outputs/dpo_rkl_a0.5

# Forward KL, alpha = 0.5
python scripts/train_dpo.py --lora_alpha -0.5 --output_dir outputs/dpo_fkl_a0.5
```

### Evaluation

```bash
# GSM8K test-set rewards of a GRPO checkpoint (writes evaluation_results.json into the checkpoint)
python scripts/eval_gsm8k_vllm.py --checkpoint_path outputs/grpo_gsm8k_rkl_a0.5

# Reward-model scores on the UltraFeedback test split
python scripts/eval_reward.py \
  --model_path outputs/dpo_rkl_a0.5 \
  --reward_model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

# Reward distribution (25 samples per prompt, first 100 prompts), for a single model or a two-model ensemble
python scripts/reward_distribution.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --model_name2 Qwen/Qwen2.5-1.5B-Instruct \
  --ensemble_alpha 0.5
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{aminian2025klregularized,
  title     = {{KL}-Regularized {RLHF} with Multiple Reference Models: Exact Solutions and Sample Complexity},
  author    = {Aminian, Gholamali and Asadi, Amir R. and Shenfeld, Idan and Mroueh, Youssef},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=j8XnFfTvXF}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). It includes code adapted from
[TRL](https://github.com/huggingface/trl) and [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math);
see [NOTICE](NOTICE) for details.
