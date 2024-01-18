#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
source /data/dreamtalk/venv/bin/activate
export SCRIPT_NAME=/dt
nohup gunicorn -w 6 --log-level debug --timeout 120 -b 0.0.0.0:9004 "app:get()" &
