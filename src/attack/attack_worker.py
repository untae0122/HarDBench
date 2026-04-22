import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import importlib.util
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from util.templates import (
    TEMPLATES_FOR_ATTACK,
    TEMPLATE_MAP_FOR_ATTACK,
    SAFETY_PROMPT,
    TEMPLATES_FOR_ATTACK_ABLATION,
    TEMPLATE_MAP_FOR_ATTACK_ABLATION,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a JSON file with OpenAI API.")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., chatgpt-4o-latest, gpt-5-2025-08-07, llama3-8b-inst, mistral-7b-inst, deepseek-R1-8b, gemini-2.0-flash)."
    )
    parser.add_argument(
        "-mp", "--model_path",
        type=str,
        default=None,
        help="Path to the model for processing. Required if model is local (llama/mistral/vicuna/deepseek)."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input JSON file to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the output JSON file. Defaults to 'attack_output_results_<timestamp>.json'."
    )
    parser.add_argument(
        "-si", "--save-interval",
        type=int,
        default=100,
        help="Number of items to process before saving interim results. Defaults to 100."
    )
    # Prompt variant: cojp (default), hq (prompt_only), safety_prompt (wrapped), ablation (custom mix)
    parser.add_argument(
        "-pv", "--prompt-variant",
        dest="prompt_variant",
        type=str,
        choices=["cojp", "hq", "safety_prompt", "ablation"],
        default="cojp",
        help=(
            "Prompt variant to use: 'cojp' (default, template with draft), 'hq' (prompt_only), "
            "'safety_prompt' (wrap base prompt), 'ablation' (customizable mix)."
        ),
    )
    # safety/ablation options
    parser.add_argument(
        "--safety-base",
        type=str,
        choices=["cojp", "hq"],
        default="cojp",
        help="When using 'safety_prompt', choose which base to wrap (cojp or hq). Default: cojp",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["base", "wo_framing", "none"],
        default="base",
        help=(
            "Ablation variant per category: 'base' (baseline), 'wo_framing' (remove framing). "
            "Uses TEMPLATES_FOR_ATTACK_ABLATION with TEMPLATE_MAP_FOR_ATTACK_ABLATION."
        ),
    )
    parser.add_argument(
        "-bs", "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing. Defaults to 1."
    )
    # Sharding for multi-GPU parallelism
    parser.add_argument(
        "--shard",
        type=int,
        default=1,
        help="Total number of shards to split dataset. Use with --shard-num."
    )
    parser.add_argument(
        "--shard-num",
        type=int,
        default=0,
        help="Which shard index to process (0-based). Used with --shard."
    )

    return parser.parse_args()


ADAPTER_FILES = {
    # Adapter file path mapping for dynamic import
    # OpenAI
    "chatgpt-4o-latest": PROJECT_ROOT / "util/models/openai/gen_openai.py",
    # Gemini
    "gemini-2.0-flash": PROJECT_ROOT / "util/models/gemini/gen_gemini.py",
    "gemini-2.5-pro": PROJECT_ROOT / "util/models/gemini/gen_gemini.py",
    # Local / HF
    "llama3-8b-inst": PROJECT_ROOT / "util/models/llama3-8b-inst/gen_llama3-8b-inst.py",
    "mistral-7b-inst": PROJECT_ROOT / "util/models/mistral-7b-inst/gen_mistral-7b-inst.py",
    # Qwen3 series
    "qwen3-8b": PROJECT_ROOT / "util/models/qwen3-8b/gen_qwen3-8b.py",
    # DeepSeek Series
    "deepseek-r1-8b": PROJECT_ROOT / "util/models/deepseek-R1-8b/gen_deepseek-R1-8b.py",
    "deepseek-r1-32b": PROJECT_ROOT / "util/models/deepseek-r1-32b/gen_deepseek-r1-32b.py",
}


def build_prompt(item: dict, args) -> str:
    """Build an attack prompt based on the selected variant.

    variants:
      - cojp: fill category-specific template with draft
      - hq:   use item['prompt_only']
      - safety_prompt: wrap base (hq|cojp) with SAFETY_PROMPT
      - ablation: use TEMPLATES_FOR_ATTACK_ABLATION per category
    """
    category = item.get('category')
    draft_text = item.get('draft', '')
    prompt_only = item.get('prompt_only', '')

    def make_cojp() -> str:
        if category in TEMPLATE_MAP_FOR_ATTACK:
            tmpl_key = TEMPLATE_MAP_FOR_ATTACK[category]
            tmpl = TEMPLATES_FOR_ATTACK.get(tmpl_key, '{draft}')
            return tmpl.format(draft=draft_text)
        return draft_text

    def make_hq() -> str:
        return prompt_only

    variant = getattr(args, 'prompt_variant', 'cojp')

    if variant == 'cojp':
        return make_cojp()
    if variant == 'hq':
        return make_hq()
    if variant == 'safety_prompt':
        base = make_hq() if getattr(args, 'safety_base', 'hq') == 'hq' else make_cojp()
        return SAFETY_PROMPT.format(prompt=base)
    if variant == 'ablation':
        mode = getattr(args, 'ablation_mode', 'base')
        suffix = {
            'base': 'baseline',
            'none': 'baseline',
            'wo_framing': 'wo_framing',
        }.get(mode, 'baseline')

        if not category:
            return make_cojp()

        map_key = f"{category}_{suffix}"
        tmpl_key = TEMPLATE_MAP_FOR_ATTACK_ABLATION.get(map_key)
        if not tmpl_key:
            return make_cojp()
        tmpl = TEMPLATES_FOR_ATTACK_ABLATION.get(tmpl_key)
        if not tmpl:
            return make_cojp()
        try:
            return tmpl.format(draft=draft_text)
        except Exception:
            return make_cojp()

    return make_cojp()


def load_adapter(model_key: str, model_path: str | None):
    adapter_path = ADAPTER_FILES.get(model_key)

    # Fallback logic for variants (e.g. safetyprompt, suba)
    if adapter_path is None:
        suffixes = ['-zeroshot', '-safetyprompt', '-suba-nb', '-suba-hq', '-suba']
        base_key = model_key
        for suffix in suffixes:
            if base_key.endswith(suffix):
                base_key = base_key[:-len(suffix)]
                break

        candidates = [base_key, f"{base_key}-instruct", f"{base_key}-inst"]
        for cand in candidates:
            if cand in ADAPTER_FILES:
                adapter_path = ADAPTER_FILES[cand]
                print(f"[*] Adapter fallback: '{model_key}' -> '{cand}'")
                break

    if adapter_path is None or not adapter_path.exists():
        known = ", ".join(sorted(ADAPTER_FILES.keys()))
        raise FileNotFoundError(f"Adapter not found for model '{model_key}'. Known: {known}")

    # Prefix module name to avoid conflicts with top-level packages (e.g. 'openai')
    module_name = f"hb_adapter_{adapter_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, str(adapter_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for adapter: {adapter_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "model_load"):
        raise AttributeError(f"Adapter module missing 'model_load': {adapter_path}")
    print(f"[+] Using adapter for model '{model_key}' ;; '{model_path}': {adapter_path}")
    effective_name = model_key if model_path is None or model_path == "" else model_path
    state = mod.model_load(model_name_or_path=effective_name)

    if not hasattr(mod, "generate"):
        raise AttributeError(f"Adapter module missing 'generate': {adapter_path}")
    return mod, state


def attacker(model_key, item, prompt, client, args, batch_size=1):
    selected_model = args.model_path or model_key

    if item.get("attack_response"):
        print(f"⏩ Already processed: {item.get('keyword', '')}")
        return item

    mod, state = client  # (module, state)
    messages = [{"role": "user", "content": prompt}]
    try:
        reply = mod.generate(
            state,
            messages,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
        )
    except Exception as e:
        print(f"❌ Inference error (single) model={model_key} keyword={item.get('keyword', '')} :: {e}")
        item["attack_prompt"] = prompt
        item["attack_model"] = selected_model
        item["attack_response"] = ""
        item["attack_error"] = str(e)
        return item

    item["attack_prompt"] = prompt
    item["attack_model"] = selected_model
    item["attack_response"] = reply
    return item


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def process_items_concurrently(client, model_key, data, save_interval, now, output_path, args):
    results = []
    count = 0
    last_saved = 0

    mod, state = client  # (module, state)
    supports_batch = bool(getattr(mod, "SUPPORTS_BATCH", True))
    shard_num = int(getattr(args, "shard_num", 0))
    print(f"batch = {args.batch_size}")
    with tqdm(total=len(data), desc="", unit="item") as pbar:
        if args.batch_size == 1 or not supports_batch:
            print("Using ThreadPoolExecutor")
            with ThreadPoolExecutor(max_workers=10) as executor:

                futures = []
                for item in data:
                    if item.get("valid_flag") is False:
                        continue
                    prompt = build_prompt(item, args)
                    futures.append(executor.submit(attacker, model_key, item, prompt, client, args))

                for future in as_completed(futures):
                    processed_item = future.result()
                    results.append(processed_item)
                    count += 1
                    pbar.update(1)

                    if count % save_interval == 0:
                        interim_file = output_path / f"attack_output_results_interim_s{shard_num}_{now}_{count}.json"
                        interim_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(interim_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"💾 Interim saved: {interim_file}")
        else:
            prompts, indices = [], []
            for idx, item in enumerate(data):
                if item.get("valid_flag") is False:
                    continue
                prompt = build_prompt(item, args)
                prompts.append(prompt)
                indices.append(idx)

            print(f"[+] Starting inference for {len(prompts)} items (batch={args.batch_size})")
            for batch_idxs in tqdm(list(chunked(range(len(prompts)), args.batch_size))):
                batch_messages = [[{"role": "user", "content": prompts[i]}] for i in batch_idxs]
                try:
                    outputs = mod.generate(
                        state,
                        batch_messages,
                        batch_size=args.batch_size,
                        max_new_tokens=2000,
                        do_sample=True,
                        temperature=0.7,
                    )
                except Exception as e:
                    print(f"❌ Batch inference error model={model_key} batch_size={len(batch_messages)} :: {e} -> fallback per-item")
                    outputs = []
                    for single in batch_messages:
                        try:
                            single_out = mod.generate(
                                state,
                                single,
                                max_new_tokens=2000,
                                do_sample=True,
                                temperature=0.7,
                            )
                        except Exception as ie:
                            print(f"  ↳ ❌ Single fallback failed :: {ie}")
                            single_out = ""
                        outputs.append(single_out)

                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch_idxs):
                    print(f"⚠️ Output length mismatch (got={len(outputs)} expected={len(batch_idxs)}) -> padding")
                    while len(outputs) < len(batch_idxs):
                        outputs.append("")

                for local_i, text in enumerate(outputs):
                    global_i = batch_idxs[local_i]
                    item_idx = indices[global_i]
                    data[item_idx]["attack_prompt"] = prompts[global_i]
                    data[item_idx]["attack_model"] = args.model_path or model_key
                    if isinstance(text, list):
                        text = text[0] if text else ""
                    data[item_idx]["attack_response"] = text if isinstance(text, str) else str(text)
                    results.append(data[item_idx])
                    count += 1
                    pbar.update(1)

                if (count - last_saved) >= save_interval:
                    interim_file = output_path / f"interim_batch_s{shard_num}.json"
                    interim_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(interim_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    last_saved = count
                    print(f"[💾] Interim saved: {interim_file} (count={count})")

    return results


def main():
    args = parse_arguments()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = PROJECT_ROOT / args.input
    if args.output is None:
        output_file = f"attack_output_results_{args.model}_{now}.json"
    else:
        output_file = PROJECT_ROOT / args.output
    output_path = Path(output_file).parent

    print("🔧 Arguments:")
    print(f"  Input file:      {input_file_path}")
    print(f"  Output file:     {output_file}")
    print(f"  Save interval:   {args.save_interval}")
    print(f"  Model:           {args.model}")
    print(f"  Model path:      {args.model_path}")
    print(f"  Prompt variant:  {args.prompt_variant}")
    print(f"  Shard: {getattr(args, 'shard', 1)} | Shard num: {getattr(args, 'shard_num', 0)}")
    print()

    print("📋 Templates:")
    for key, value in TEMPLATES_FOR_ATTACK.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------")
    for key, value in TEMPLATE_MAP_FOR_ATTACK.items():
        print(f"  {key}: {value}")
    print()

    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format: {input_file_path}")
        return

    # Shard slicing: [shard_num::shard]
    total_shard = max(1, int(getattr(args, "shard", 1)))
    shard_num = max(0, int(getattr(args, "shard_num", 0)))
    if shard_num >= total_shard:
        print(f"❌ shard_num({shard_num}) must be < shard({total_shard}).")
        return

    # Inject original index for later merging
    indexed_data = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            new_item = dict(item)
        else:
            continue
        new_item.setdefault("orig_index", idx)
        indexed_data.append(new_item)

    shard_data = indexed_data[shard_num::total_shard]
    print(f"[+] Shard slicing -> total={len(indexed_data)}, shard={total_shard}, shard_num={shard_num}, this_shard_size={len(shard_data)}")

    # Load adapter and run attack
    client = load_adapter(args.model, args.model_path)
    results = process_items_concurrently(client, args.model, shard_data, args.save_interval, now, output_path, args)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Saved {len(results)} results: {output_file}")
    error_count = sum(1 for r in results if (not r.get("attack_response")) and r.get("attack_prompt"))
    if error_count:
        print(f"⚠️ Inference errors (skipped) count={error_count} (see per-item attack_error if present)")


if __name__ == "__main__":
    main()