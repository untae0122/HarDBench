#!/usr/bin/env bash

# Minimal GPT attack runner (background + log)
LOG_FILE="attack_chatgpt_$(date +'%Y%m%d_%H%M%S').log"

echo "[$(date)] start" >> "$LOG_FILE"
nohup python3 attack_worker.py -m chatgpt-4o-latest -i 0_data/train_dataset_benign_1455.json >> "$LOG_FILE" 2>&1 &
echo "[$(date)] PID: $!" >> "$LOG_FILE"