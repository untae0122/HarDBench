#!/usr/bin/env python3
"""
Pattern management utility.
Analyzes unmatched_patterns.log to suggest and test new regex patterns for the compare evaluator.
"""

import re
import json
from collections import Counter
from pathlib import Path
from datetime import datetime


def analyze_unmatched_patterns(log_file="unmatched_patterns.log"):
    """Analyze unmatched reply patterns and suggest candidates."""
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into individual log entries
    entries = content.split("-" * 80)

    replies = []
    for entry in entries:
        if "Reply:" in entry:
            reply_part = entry.split("Reply:")[1].strip()
            if reply_part:
                replies.append(reply_part)

    print(f"Replies to analyze: {len(replies)}")
    print("\n" + "=" * 50)
    print("Pattern analysis of unmatched replies")
    print("=" * 50)

    # Find common patterns
    potential_patterns = []

    for reply in replies:
        print(f"\nReply: {reply[:200]}...")

        # Find lines that contain A or B
        lines = reply.split('\n')
        for line in lines:
            if re.search(r'\b[AB]\b', line, re.IGNORECASE):
                print(f"  -> Candidate line: {line.strip()}")
                potential_patterns.append(line.strip())

    # Group similar patterns
    print(f"\n\nCandidate patterns ({len(potential_patterns)}):")
    pattern_counter = Counter(potential_patterns)

    for pattern, count in pattern_counter.most_common(10):
        print(f"  {count}x: {pattern}")

    return potential_patterns


def suggest_new_patterns(potential_patterns):
    """Convert candidate patterns into regex suggestions."""
    print("\n\nSuggested new regex patterns:")
    print("-" * 40)

    suggestions = []

    for pattern in set(potential_patterns):
        # Heuristic: convert raw text into a regex
        escaped = re.escape(pattern)
        # Replace A/B literals with a capture group
        regex_pattern = escaped.replace(r'A', r'(A|B)').replace(r'B', r'(A|B)')
        # Clean up redundant groups
        regex_pattern = re.sub(r'\(A\|B\)\|\(A\|B\)', r'(A|B)', regex_pattern)
        # Make whitespace flexible
        regex_pattern = re.sub(r'\\ ', r'\\s*', regex_pattern)

        if regex_pattern not in suggestions:
            suggestions.append(regex_pattern)
            print(f"  r\"{regex_pattern}\",")

    return suggestions


def test_patterns_on_samples(patterns, sample_replies):
    """Test proposed patterns against a sample of replies."""
    print("\n\nPattern test results:")
    print("-" * 40)

    for i, pattern in enumerate(patterns[:5]):  # Test first 5 patterns only
        print(f"\nPattern {i + 1}: {pattern}")
        matches = 0
        for reply in sample_replies[:10]:  # Test on first 10 samples only
            if re.search(pattern, reply, re.IGNORECASE):
                matches += 1
                match = re.search(pattern, reply, re.IGNORECASE)
                if match:
                    print(f"  ✅ Match: {match.group(1) if match.groups() else match.group(0)}")

        print(f"  Total matches: {matches}/{len(sample_replies[:10])}")


def backup_current_patterns():
    """Back up the current pattern configuration to a timestamped JSON file."""
    from pattern_config import DANGEROUS_ANSWER_PATTERNS

    backup_file = f"pattern_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "patterns": DANGEROUS_ANSWER_PATTERNS
    }

    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)

    print(f"Current patterns backed up to: {backup_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pattern management utility for HarDBench compare evaluator")
    parser.add_argument("--analyze", action="store_true", help="Analyze unmatched patterns from log")
    parser.add_argument("--backup", action="store_true", help="Backup current pattern config")
    parser.add_argument("--log-file", default="unmatched_patterns.log", help="Path to unmatched patterns log")

    args = parser.parse_args()

    if args.backup:
        backup_current_patterns()

    if args.analyze:
        potential_patterns = analyze_unmatched_patterns(args.log_file)
        if potential_patterns:
            suggestions = suggest_new_patterns(potential_patterns)
            test_patterns_on_samples(suggestions, potential_patterns)

            print("\n\nTo add patterns to pattern_config.py:")
            print("1. Backup current patterns: python pattern_manager.py --backup")
            print("2. Add suggested patterns to DANGEROUS_ANSWER_PATTERNS in pattern_config.py")
            print("3. Test and restore from backup if needed")


if __name__ == "__main__":
    main()
