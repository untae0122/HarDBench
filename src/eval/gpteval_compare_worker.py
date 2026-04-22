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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_openai_available = True
try:
    from openai import OpenAI
except Exception:
    _openai_available = False

_gemini_available = True
try:
    from google import genai  # new layout
except Exception:
    try:
        import google.generativeai as genai  # legacy
    except Exception:
        _gemini_available = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from util.templates import EVAL_PROMPT, BENIGN_EVAL_PROMPT, COMPARE_EVAL_PROMPT
from pattern_config import DANGEROUS_ANSWER_PATTERNS, SCORE_PATTERNS, YES_NO_PATTERNS


def extract_dangerous_answer_score(reply):
    """
    Extract which answer (A or B) is more dangerous.
    Uses regex patterns defined in pattern_config.
    """
    for pattern in DANGEROUS_ANSWER_PATTERNS:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Return None if nothing matched
    return None


def extract_score(reply):
    """Extract numeric score (unused in compare but kept for parity)."""
    for pattern in SCORE_PATTERNS:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_yes_no_answer(reply):
    """Extract a Yes/No style answer (legacy utility)."""
    for pattern in YES_NO_PATTERNS:
        match = re.search(pattern, reply, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def log_unmatched_pattern(item, reply, log_file="unmatched_patterns.log"):
    """
    Log replies whose pattern was not matched for future pattern improvements.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Keyword: {item.get('keyword', 'N/A')}\n")
        f.write(f"Reply: {reply}\n")
        f.write("-" * 80 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="HarDBench compare evaluation worker (OpenAI + Gemini)")
    parser.add_argument('-i','--input', type=str, required=True, help='Input JSON path')
    parser.add_argument('-o','--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('-si','--save-interval', type=int, default=100, help='Interim save interval')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06', help='Model name to use')
    parser.add_argument('--provider', type=str, choices=['auto','openai','gemini'], default='auto', help='Force provider (auto = infer)')
    parser.add_argument('--max-workers', type=int, default=10, help='Thread pool size')
    parser.add_argument('--dry-run', action='store_true', help='Test run without API calls')
    return parser.parse_args()

def detect_provider(model_name: str, explicit: str) -> str:
    if explicit != 'auto':
        return explicit
    if 'gemini' in model_name.lower():
        return 'gemini'
    return 'openai'

def build_clients(provider: str, model_name: str):
    openai_client = None
    gemini_client = None
    if provider == 'openai':
        if not _openai_available:
            raise RuntimeError('openai package not installed')
        if not OPENAI_API_KEY:
            raise RuntimeError('OPENAI_API_KEY not set')
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        if not _gemini_available:
            raise RuntimeError('google-generativeai package not installed')
        if not GOOGLE_API_KEY:
            raise RuntimeError('GOOGLE_API_KEY not set')
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    return openai_client, gemini_client

def call_model(provider: str, model_name: str, question: str, openai_client=None, gemini_client=None, dry_run=False):
    if dry_run:
        return f"[DRY-RUN] {model_name}: {question[:50]}..."
    if provider == 'openai':
        resp = openai_client.chat.completions.create(model=model_name, messages=[{"role":"user","content":question}])
        return resp.choices[0].message.content
    else:
        resp = gemini_client.models.generate_content(model=model_name, contents=question)
        return getattr(resp, 'text', str(resp))


def eval_score_api(model, provider, item, openai_client, gemini_client, args):
    score = 0
    reply = ""

    if "compare" in item:
        try:
            if int(item["compare"]) != 0:
                print(f"⏩ Already processed: {item.get('keyword')}")
                return reply , item
        except Exception:
            pass
    try:
        if int(item.get("score",0)) != 5:
            return reply, item
    except Exception:
        return reply, item

    try:
        if item.get("harmful", True):
            question = COMPARE_EVAL_PROMPT.format(answer_a=item.get("attack_response",""), answer_b=item.get("draft",""))
        else:
            # Non harmful path: still construct compare but semantics might differ
            question = COMPARE_EVAL_PROMPT.format(answer_a=item.get("attack_response",""), answer_b=item.get("draft",""))
        reply = call_model(provider, model, question, openai_client, gemini_client, args.dry_run)
        score = extract_dangerous_answer_score(reply)
        if score is None:
            print(f"⚠️  Pattern match failed ({item.get('keyword')}): {reply[:100]}...")
            log_unmatched_pattern(item, reply)
            item["parsing_failed"] = True
            item["raw_reply"] = reply
        else:
            print(f"Answer: {score}")
            if score == "A":
                item["moreharmful"] = "attack_response"
            elif score == "B":
                item["moreharmful"] = "draft"
            item["compare"] = 1
            item["scoring_model"] = model
            item["scoring_provider"] = provider
    except Exception as e:
        print(f"❌ Error ({item.get('keyword')}): {e}")
    return reply , item


def process_items_concurrently(openai_client, gemini_client, provider, model, data, save_interval, now, output_path, args):
    """Process items concurrently with thread pool."""
    results = []
    reply_results = []
    count = 0

    with tqdm(total=len(data), desc="Processing", unit="item") as pbar:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    eval_score_api,
                    model,
                    provider,
                    item,
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
                    if count % save_interval == 0:
                        interim_file = output_path / f"eval_interim_{now}_{count}.json"
                        with open(interim_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"💾 Interim save: {interim_file}")
                except Exception as e:
                    print(f"❌ Error: {e}")

    return results, reply_results

def main():
    args = parse_arguments()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = PROJECT_ROOT / args.input
    if args.output is None:
        output_file = f"attack_output_results_compare_eval_{now}.json"
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
    print(f"  Workers: {args.max_workers}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Dry-run: {args.dry_run}")
    print("---------------------------------------------")
    print("📋 Compare evaluation prompt (head):\n", COMPARE_EVAL_PROMPT[:400], '...')

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

    results, reply_results = process_items_concurrently(openai_client, gemini_client, provider, args.model, data, args.save_interval, now, output_path, args)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_path / f"eval_reply_{now}.json", 'w', encoding='utf-8') as f:
        json.dump(reply_results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Saved results:\n - Scored: {output_file}\n - Raw replies: {output_path / f'eval_reply_{now}.json'}")



if __name__ == "__main__":
    main()