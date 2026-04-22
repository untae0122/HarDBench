# HarDBench

### This is the official project of the paper:
[ACL 2026] [HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human–LLM Collaborative Writing](https://arxiv.org/abs/2604.19274v1)

---

## Overview

HarDBench is a standalone evaluation pipeline for assessing LLM safety against draft-based co-authoring jailbreak attacks. The pipeline follows four sequential steps:

```
Attack  →  Eval  →  Compare  →  Score
```

| Step    | Script                                    | Description |
|---------|-------------------------------------------|-------------|
| Attack  | `src/attack/attack_worker.py`             | Send attack prompts to the target model, collect `attack_response` |
| Eval    | `src/eval/gpteval_worker.py`              | Score responses with GPT/Gemini (1–5) |
| Compare | `src/eval/gpteval_compare_worker.py`      | For score-5 items, compare `attack_response` vs draft harmfulness |
| Score   | `src/eval/cal_score.py`                   | Compute ASR (score 4–5 ratio) and RAR, save JSON summary |

> **Note**: The `hq` prompt variant (`prompt_variant: hq`) automatically skips the Compare step.

---

## Directory Structure

```
HarDBench/
├── run_experiment.py               # Main pipeline runner
├── .env                            # API key configuration
├── data/                           # ← Place dataset files here
│   ├── HarDbench_test.json         # Test dataset
│   ├── HarDbench_train.json        # Train dataset
│   └── HarDbench_all.json          # Full dataset
├── configs/
│   ├── models/                     # Per-model YAML configs
│   └── experiment_example.yaml     # Example experiment config
├── src/
│   ├── attack/
│   │   ├── attack_worker.py        # [Step 1] Attack generation
│   │   └── merge_shard.py          # Merge multi-GPU shard outputs
│   ├── eval/
│   │   ├── gpteval_worker.py       # [Step 2] GPT-based scoring (1–5)
│   │   ├── gpteval_compare_worker.py  # [Step 3] attack vs draft comparison
│   │   ├── cal_score.py            # [Step 4] ASR / RAR calculation
│   │   ├── pattern_config.py
│   │   └── pattern_manager.py
│   └── util/
│       ├── templates.py            # Prompt templates
│       └── models/                 # Model adapters (OpenAI, Gemini, LLaMA3, Mistral, etc.)
└── results/                        # Experiment results (auto-created)
```

---

## Dataset

### 📦 Download the Dataset

Download the dataset from Hugging Face:  
👉 [https://huggingface.co/datasets/untae/HarDBench](https://huggingface.co/datasets/untae/HarDBench)

### 🔽 Place Dataset Files

After downloading, place the JSON files under the `data/` directory:

```
HarDBench/
└── data/
    ├── HarDbench_test.json      # used for evaluation
    ├── HarDbench_train.json
    └── HarDbench_all.json
```

The `input` field in your experiment config should point to the file you want to evaluate, e.g.:

```yaml
pipelines:
  - prompt_variant: "cojp"
    input: "data/HarDbench_test.json"
```

### 💬 Dataset Splits

| File | Split | Description |
|------|-------|-------------|
| `HarDbench_train.json` | train | Training split |
| `HarDbench_test.json`  | test  | Evaluation split (recommended) |
| `HarDbench_all.json`   | all   | Full dataset |

---

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/your-org/HarDBench.git
cd HarDBench
pip install -r requirements.txt
```

---

## Quick Start

### 1. Configure `.env`

```env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY        # Required for Gemini models
HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN  # Required for gated HF models
```

### 2. Write a config

Copy `configs/experiment_example.yaml` and edit `model.name` and `model.path`:

```yaml
model:
  name: "llama3-8b-inst"
  path: "/path/to/model"

pipelines:
  - prompt_variant: "cojp"
    input: "data/HarDbench_test.json"
    save_interval: 50
    batch_size: 1
```

### 3. Run

```bash
cd /path/to/HarDBench
python run_experiment.py --config configs/experiment_example.yaml --gpu 0
```

### Multi-GPU (Sharding)

```bash
# Run each shard on a separate GPU
CUDA_VISIBLE_DEVICES=0 python src/attack/attack_worker.py \
  --model llama3-8b-instruct --model_path /path/to/model \
  --input data/HarDbench_test.json --output results/shard_0.json \
  --shard 2 --shard-num 0

CUDA_VISIBLE_DEVICES=1 python src/attack/attack_worker.py \
  --model llama3-8b-instruct --model_path /path/to/model \
  --input data/HarDbench_test.json --output results/shard_1.json \
  --shard 2 --shard-num 1

# Merge shards
python src/attack/merge_shard.py \
  -i results/shard_0.json results/shard_1.json \
  -o results/attack_merged.json
```

---

## Supported Models

| Model Key | Type |
|-----------|------|
| `chatgpt-4o-latest` | API |
| `gemini-2.0-flash`, `gemini-2.5-pro` | API |
| `llama3-8b-inst` | Local |
| `mistral-7b-inst` | Local |
| `deepseek-R1-8b`, `deepseek-r1-32b` | Local |
| `qwen3-8b`, `qwen3-14b`, `qwen3-32b`, ... | Local |

---

## Output Structure

```
results/<model>/<experiment>_<timestamp>/
├── artifacts/
│   ├── attack_output_cojp.json     # Raw attack responses
│   ├── eval_output_cojp.json       # Scoring results (1–5)
│   ├── compare_output_cojp.json    # Comparison results
│   └── score_output_cojp.json      # Final ASR / RAR summary
└── logs/
    ├── attack_cojp.log
    ├── eval_cojp.log
    └── ...
```

---

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@inproceedings{hardbench2026,
  title     = {},
  author    = {},
  booktitle = {},
  year      = {2026}
}
```
