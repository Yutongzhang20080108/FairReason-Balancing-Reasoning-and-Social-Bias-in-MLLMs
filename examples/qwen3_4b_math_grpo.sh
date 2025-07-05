#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/workspace/Qwen3-1.7B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_1.7b_math_grpo
