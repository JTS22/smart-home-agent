#!/bin/bash
export UVICORN_PORT=$1
export MLFLOW_EXPERIMENT_NAME=$2
.venv/bin/python main.py