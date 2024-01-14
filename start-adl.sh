#!/bin/bash
TURBO=/root/autodl-tmp/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
eval "$(conda shell.bash hook)"
conda activate /root/autodl-tmp/conda/dreamtalk
export SCRIPT_NAME=/dt
nohup gunicorn -w 6 --log-level debug --timeout 120 -b 0.0.0.0:8004 "app:get()" &
