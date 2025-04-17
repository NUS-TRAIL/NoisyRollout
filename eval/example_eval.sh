#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate noisyrollout

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export GOOGLE_API_KEY="xxx"

# Define list of model paths to evaluate
HF_MODEL_PATH=""
RESULTS_DIR="results/"
EVAL_DIR="~/NoisyRollout/eval"
DATA_DIR="~/NoisyRollout/eval/data"

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

cd $EVAL_DIR
python main.py \
  --model $HF_MODEL_PATH \
  --output-dir $RESULTS_DIR \
  --data-path $DATA_DIR \
  --datasets geo3k,hallubench,mathvista,wemath,mathverse,mathvision \
  --tensor-parallel-size 2 \
  --system-prompt="$SYSTEM_PROMPT" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --eval-threads 24