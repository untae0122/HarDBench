import json
import re
import sys
import argparse
import random
import time
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini key (must be set in environment)

# Lazy import (only when needed) -> OpenAI
_openai_available = True
try:
    from openai import OpenAI
except Exception:
    _openai_available = False

# Gemini library (optional)
_gemini_available = True
try:
    from google import genai  # new package layout
except Exception:
    try:
        import google.generativeai as genai  # legacy layout
    except Exception:
        _gemini_available = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from util.templates import EVAL_PROMPT, BENIGN_EVAL_PROMPT


def parse_arguments():
    parser = argparse.ArgumentParser(description="HarDBench evaluation worker (OpenAI GPT + optional Gemini support)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON path")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path (default attack_output_results_eval_<ts>.json)")
    parser.add_argument("-si", "--save-interval", type=int, default=100, help="Save interval for interim results (default 100)")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model name (e.g. gpt-4o-2024-08-06, gemini-1.5-pro, gemini-1.5-flash)")
    parser.add_argument("--provider", type=str, choices=["auto","openai","gemini"], default="auto", help="Force provider (auto: infer from model name)")
    parser.add_argument("--max-workers", type=int, default=10, help="ThreadPoolExecutor max_workers (default 10)")
    parser.add_argument("--dry-run", action="store_true", help="Run without API calls (debug flow)")
    return parser.parse_args()


def detect_provider(model_name: str, explicit: str) -> str:
    if explicit != 'auto':
        return explicit
    lower = model_name.lower()
    if 'gemini' in lower:
        return 'gemini'
    return 'openai'


def build_clients(provider: str, model_name: str):
    openai_client = None
    gemini_client = None
    if provider == 'openai':
        if not _openai_available:
            raise RuntimeError("Cannot find 'openai' package. Please 'pip install openai' and retry.")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == 'gemini':
        if not _gemini_available:
            raise RuntimeError("Cannot find 'google-generativeai' package. Please 'pip install google-generativeai'.")
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
        # New client style
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return openai_client, gemini_client


def call_model(provider: str, model_name: str, question: str, openai_client=None, gemini_client=None, dry_run=False) -> str:
    if dry_run:
        return f"[DRY-RUN] {model_name}: {question[:60]}..."
    if provider == 'openai':
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
    elif provider == 'gemini':
        # Using new client API
        resp = gemini_client.models.generate_content(model=model_name, contents=question)
        return getattr(resp, 'text', str(resp))
    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_score(reply: str):
    # Same parsing rules as original logic
    if "#thescore:" in reply:
        m = re.search(r"#thescore:\s*(\d+)", reply)
        if m: return m.group(1)
    if "#thescore" in reply:
        m = re.search(r"#thescore\s*(\d+)", reply)
        if m: return m.group(1)
    if '#theanswer:' in reply:
        m = re.search(r"#theanswer:\s*(yes|no)", reply, re.IGNORECASE)
        if m: return m.group(1).upper()
    if '#theanswer' in reply:
        m = re.search(r"#theanswer\s*(yes|no)", reply, re.IGNORECASE)
        if m: return m.group(1).upper()
    return 0


def eval_score_api(model, provider, item, prompt, openai_client, gemini_client, args):
    score = 0
    reply = ""
    # Skip if already scored
    if "scoring_model" in item and "score" in item:
        try:
            if str(item["score"]) not in ("0", "", "None"):
                print(f"⏩ Already processed: {item.get('keyword','?')}")
                return reply, item
        except Exception:
            pass
    try:
        if item.get("harmful", True):
            question = EVAL_PROMPT.format(response=prompt)
        else:
            question = BENIGN_EVAL_PROMPT.format(user_input=item.get("attack_response",""), model_output=prompt)
        reply = call_model(provider, model, question, openai_client, gemini_client, dry_run=args.dry_run)
        score = parse_score(reply)
        item["scoring_model"] = model
        item["scoring_provider"] = provider
        item["score"] = score
    except Exception as e:
        item["scoring_model"] = model
        item["scoring_provider"] = provider
        item["score"] = score
        item["error"] = str(e)
        print(f"❌ Error ({item.get('keyword','?')}): {e}")
    return reply, item


def process_items_concurrently(openai_client, gemini_client, provider, model, data, save_interval, now, output_path, args):
    results = []
    reply_results = []
    count = 0
    max_workers = args.max_workers
    with tqdm(total=len(data), desc="Processing", unit="item") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    eval_score_api,
                    model,
                    provider,
                    item,
                    item.get("attack_response",""),
                    openai_client,
                    gemini_client,
                    args
                ) for item in data
            ]
            for future in as_completed(futures):
                try:
                    reply, processed_item = future.result()
                    results.append(processed_item)
                    reply_results.append({
                        "keyword": processed_item.get("keyword"),
                        "scoring_model": model,
                        "provider": provider,
                        "reply": reply
                    })
                    count += 1
                    pbar.update(1)
                    if save_interval > 0 and count % save_interval == 0:
                        interim_file = output_path / f"eval_interim_{now}_{count}.json"
                        with open(interim_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"💾 Interim save: {interim_file}")
                except Exception as e:
                    print(f"❌ Error (collect loop): {e}")
    return results, reply_results

def main():
    args = parse_arguments()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = PROJECT_ROOT / args.input
    if args.output is None:
        output_file = f"attack_output_results_eval_{now}.json"
    else:
        output_file = PROJECT_ROOT / args.output
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    provider = detect_provider(args.model, args.provider)

    print("🔧 Run config:")
    print(f"  Input: {input_file_path}")
    print(f"  Output: {output_file}")
    print(f"  Model: {args.model}")
    print(f"  Provider: {provider}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Dry-run: {args.dry_run}")
    print("--------------------------------------------------")
    print("📋 Evaluation prompt template (head):\n", EVAL_PROMPT[:400], '...')

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format: {input_file_path}")
        return

    try:
        openai_client, gemini_client = build_clients(provider, args.model)
    except Exception as e:
        print(f"❌ Client init failed: {e}")
        return

    results, reply_results = process_items_concurrently(
        openai_client, gemini_client, provider, args.model, data, args.save_interval, now, output_path, args
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_path / f"eval_reply_{now}.json", 'w', encoding='utf-8') as f:
        json.dump(reply_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Saved results:\n - Scored: {output_file}\n - Raw replies: {output_path / f'eval_reply_{now}.json'}")



if __name__ == "__main__":
    main()