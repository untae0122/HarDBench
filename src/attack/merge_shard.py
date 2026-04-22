#!/usr/bin/env python
import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Merge sharded attack outputs into one ordered file.")
    p.add_argument("-i", "--inputs", nargs="+", required=True, help="List of shard JSON files to merge")
    p.add_argument("-o", "--output", required=True, help="Path to write merged JSON")
    p.add_argument("--expect-total", type=int, default=None, help="Optional expected total item count for validation")
    return p.parse_args()


def main():
    args = parse_args()
    merged = {}
    total_read = 0

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            raise FileNotFoundError(f"Shard file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in shard {path}: {e}")
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("orig_index")
            if idx is None:
                # If missing, append at the end with a large index offset
                idx = 10**12 + len(merged)
            # If duplicate index occurs, keep the first seen (or replace? choose keep first)
            if idx not in merged:
                merged[idx] = item
        total_read += len(data)

    # Order by original index
    ordered = [merged[k] for k in sorted(merged.keys())]

    # Optional validation
    if args.expect_total is not None:
        if len(ordered) != args.expect_total:
            print(f"[WARN] Merged count {len(ordered)} != expected {args.expect_total}. Shards may be incomplete.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print(f"[+] Merged {len(ordered)} items from {len(args.inputs)} shards -> {out_path}")


if __name__ == "__main__":
    main()
