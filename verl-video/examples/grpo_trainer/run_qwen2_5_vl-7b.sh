set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HUB_OFFLINE=1
export HOME=/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl-video
export PYTORCH_CUDA_ALLOC_CONF=""
export NNODES=${NNODES:-1}
CKPTS_DIR="/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/cpkts"
WORKLOAD_RUN_ID=${WORKLOAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export WORKLOAD_COLLECT_LENGTHS=${WORKLOAD_COLLECT_LENGTHS:-1}
export WORKLOAD_LENGTH_DIR=${WORKLOAD_LENGTH_DIR:-/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/logs/packed_length_log/${WORKLOAD_RUN_ID}}
mkdir -p "${WORKLOAD_LENGTH_DIR}"
echo "Packed workload length logs: ${WORKLOAD_LENGTH_DIR}"
# export WANDB_API_KEY=cac422113ce38ff90b619824d9cb1277b557f082
# wandb login cac422113ce38ff90b619824d9cb1277b557f082

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/data/Processed-Cosmos-Reason1-RL-Dataset/train_robovqa_4.parquet \
    data.val_files=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/data/Processed-Cosmos-Reason1-RL-Dataset/test_robovqa_4.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_response_length=2048 \
    data.image_key=images \
    \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/models/Qwen2.5-VL-7B-COT-SFT \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32000 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32000 \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_video-r1' \
    trainer.experiment_name='qwen2_5_vl_7b_cosmos_test-16k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.resume_mode=disable \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=10 $@