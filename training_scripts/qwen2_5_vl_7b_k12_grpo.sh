#!/bin/bash
set -x
source ~/.bashrc
source ~/miniconda3/bin/activate noisyrollout
cd ~/NoisyRollout

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_API_KEY="xxx"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
EXPERIMENT_NAME=qwen2_5_vl_7b_k12_grpo
PROJECT_NAME=noisy_rollout
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=training_scripts/config.yaml \
    data.train_files=xyliu6/k12-freeform@mini_train \
    data.val_files=xyliu6/k12-freeform@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.max_response_length=2048 \
    data.max_pixels=1000000 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.use_kl_loss=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.reward.compute_score=math \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.n=12 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=20 \
    worker.actor.is_noisy=false