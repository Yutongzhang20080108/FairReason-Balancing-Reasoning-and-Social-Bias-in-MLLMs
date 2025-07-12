#!/bin/bash

# Required for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="/workspace/BRSS-Balancing-Reasoning-with-Social-Bias/model/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=30000,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:30000,temperature:0.6,top_p:0.95}"
OUTPUT_DIR="/workspace/BRSS-Balancing-Reasoning-with-Social-Bias/data/eval/$MODEL"

TASK="aime24"
#TASK="math_500"
#TASK="gpqa:diamond"


lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"

echo "LightEval evaluation completed. Results saved to $OUTPUT_DIR"