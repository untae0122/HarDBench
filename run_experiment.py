#!/usr/bin/env python3
"""
HarDBench Experiment Runner
Runs only the core Attack → Eval → Compare → Score pipeline.
Fine-tuning and external benchmarks are excluded.
"""
import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # If model_config key is present, load and deep-merge the referenced YAML
    if 'model_config' in config:
        base_path = PROJECT_ROOT / config['model_config']
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)

            def deep_merge(base, update):
                for k, v in update.items():
                    if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                        deep_merge(base[k], v)
                    else:
                        base[k] = v
                return base

            deep_merge(base_config, config)
            return base_config
        else:
            print(f"Warning: Model config {base_path} not found.")

    return config


def run_command(cmd, log_file=None, env=None, cwd=None):
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"  (in directory: {cwd})")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    else:
        subprocess.check_call(cmd, env=env, cwd=cwd)


def run_pipeline(pipeline_config, config, env, result_dir):
    """
    Run a single pipeline: Attack → Eval → Compare → Score.
    """
    artifacts_dir = result_dir / "artifacts"
    logs_dir = result_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{pipeline_config.get('prompt_variant', 'default')}"

    attack_output  = artifacts_dir / f"attack_output{suffix}.json"
    eval_output    = artifacts_dir / f"eval_output{suffix}.json"
    compare_output = artifacts_dir / f"compare_output{suffix}.json"
    score_output   = artifacts_dir / f"score_output{suffix}.json"

    # ── Step 1: Attack ────────────────────────────────────────────────────────
    if attack_output.exists():
        print(f"[*] Attack output already exists: {attack_output}. Skipping.")
    else:
        print(f"\n🚀 [1/4] Attack ({suffix.strip('_')}) ...")
        attack_cmd = [
            sys.executable, str(PROJECT_ROOT / "src/attack/attack_worker.py"),
            "--model",         config['model']['name'],
            "--input",         str(PROJECT_ROOT / pipeline_config['input']),
            "--output",        str(attack_output),
            "--save-interval", str(pipeline_config.get('save_interval', 50)),
            "--batch-size",    str(pipeline_config.get('batch_size', 1)),
            "--prompt-variant", pipeline_config.get('prompt_variant', 'cojp'),
        ]

        if config['model'].get('path'):
            attack_cmd.extend(["--model_path", config['model']['path']])

        if 'ablation_mode' in pipeline_config:
            attack_cmd.extend(["--ablation-mode", pipeline_config['ablation_mode']])

        if 'attack' in config and 'extra_args' in config['attack']:
            attack_cmd.extend(config['attack']['extra_args'])

        if 'env' in pipeline_config:
            attack_cmd = ["conda", "run", "-n", pipeline_config['env'], "--no-capture-output"] + attack_cmd

        try:
            run_command(attack_cmd, log_file=logs_dir / f"attack{suffix}.log", env=env)
            print(f"✅ Attack done: {attack_output}")
        except subprocess.CalledProcessError:
            print(f"❌ Attack failed. Log: {logs_dir / f'attack{suffix}.log'}")
            return

    # ── Step 2: Eval ──────────────────────────────────────────────────────────
    if eval_output.exists():
        print(f"[*] Eval output already exists: {eval_output}. Skipping.")
    else:
        print(f"\n🚀 [2/4] Eval ({suffix.strip('_')}) ...")
        eval_cmd = [
            sys.executable, str(PROJECT_ROOT / "src/eval/gpteval_worker.py"),
            "--input",         str(attack_output),
            "--output",        str(eval_output),
            "--save-interval", str(pipeline_config.get('save_interval', 50)),
        ]

        if 'eval' in config and 'extra_args' in config['eval']:
            eval_cmd.extend(config['eval']['extra_args'])

        if 'env' in pipeline_config:
            eval_cmd = ["conda", "run", "-n", pipeline_config['env'], "--no-capture-output"] + eval_cmd

        try:
            run_command(eval_cmd, log_file=logs_dir / f"eval{suffix}.log", env=env)
            print(f"✅ Eval done: {eval_output}")
        except subprocess.CalledProcessError:
            print(f"❌ Eval failed. Log: {logs_dir / f'eval{suffix}.log'}")

    # ── Step 3: Compare ───────────────────────────────────────────────────────
    # Skip Compare step for HQ variant
    if pipeline_config.get('prompt_variant') == 'hq':
        print(f"[*] Skipping Compare for HQ variant.")
    elif compare_output.exists():
        print(f"[*] Compare output already exists: {compare_output}. Skipping.")
    else:
        print(f"\n🚀 [3/4] Compare ({suffix.strip('_')}) ...")
        compare_cmd = [
            sys.executable, str(PROJECT_ROOT / "src/eval/gpteval_compare_worker.py"),
            "--input",         str(eval_output),
            "--output",        str(compare_output),
            "--save-interval", str(pipeline_config.get('save_interval', 50)),
        ]

        if 'compare' in config and 'extra_args' in config['compare']:
            compare_cmd.extend(config['compare']['extra_args'])

        if 'env' in pipeline_config:
            compare_cmd = ["conda", "run", "-n", pipeline_config['env'], "--no-capture-output"] + compare_cmd

        try:
            run_command(compare_cmd, log_file=logs_dir / f"compare{suffix}.log", env=env)
            print(f"✅ Compare done: {compare_output}")
        except subprocess.CalledProcessError:
            print(f"❌ Compare failed. Log: {logs_dir / f'compare{suffix}.log'}")

    # ── Step 4: Score ─────────────────────────────────────────────────────────
    if score_output.exists():
        print(f"[*] Score output already exists: {score_output}. Skipping.")
    else:
        print(f"\n🚀 [4/4] Cal Score ({suffix.strip('_')}) ...")
        cal_input = str(compare_output) if compare_output.exists() else str(eval_output)
        cal_cmd = [
            sys.executable, str(PROJECT_ROOT / "src/eval/cal_score.py"),
            "--input",     cal_input,
            "--output",    str(score_output),
            "--reference", str(PROJECT_ROOT / pipeline_config['input']),
        ]

        if 'cal' in config and 'extra_args' in config['cal']:
            cal_cmd.extend(config['cal']['extra_args'])

        if 'env' in pipeline_config:
            cal_cmd = ["conda", "run", "-n", pipeline_config['env'], "--no-capture-output"] + cal_cmd

        try:
            run_command(cal_cmd, log_file=logs_dir / f"cal{suffix}.log", env=env)
            print(f"✅ Cal Score done: {score_output}")
        except subprocess.CalledProcessError:
            print(f"❌ Cal Score failed. Log: {logs_dir / f'cal{suffix}.log'}")


def main():
    parser = argparse.ArgumentParser(description="HarDBench Pipeline Runner (Attack → Eval → Compare → Score)")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment YAML config file.")
    parser.add_argument("--gpu",    type=str, default="0",  help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1').")
    args = parser.parse_args()

    # 1. Load config
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # 2. Setup env
    load_dotenv()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    if "openai_api_key" in config:
        env["OPENAI_API_KEY"] = config["openai_api_key"]
    elif "OPENAI_API_KEY" not in env:
        print("⚠️  Warning: OPENAI_API_KEY not set. Eval steps may fail.")

    # 3. Results directory
    model_name      = config['model']['name']
    experiment_name = config.get('experiment_name', config_path.stem)
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")

    if 'output_dir' in config:
        result_dir = PROJECT_ROOT / config['output_dir']
    else:
        result_dir = PROJECT_ROOT / "results" / model_name / f"{experiment_name}_{timestamp}"

    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Results directory: {result_dir}")

    # 4. Run pipelines
    pipelines = config.get('pipelines', [config['pipeline']] if 'pipeline' in config else [])

    if not pipelines:
        print("❌ No pipeline configuration found in config.")
        sys.exit(1)

    print(f"\n🚀 Running {len(pipelines)} pipeline(s) ...")
    for p_config in pipelines:
        run_pipeline(p_config, config, env, result_dir)

    print(f"\n🎉 All done! Results in: {result_dir}")


if __name__ == "__main__":
    main()
