#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
if [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi
nohup gunicorn -w 6 --log-level debug --timeout 120 -b 0.0.0.0:8004 "app:get()" &
