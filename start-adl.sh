#!/bin/bash
TURBO=/root/autodl-tmp/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
conda activate /root/autodl-tmp/conda/dreamtalk
nohup gunicorn -w 6 --log-level debug --timeout 120 -b 0.0.0.0:8004 "app:get()" &
