# run on 8xH100
# make sure your current working directory is the root of the project

PROJECT_DIR="/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# export http_proxy="https://zhoujiecheng:SxtTJSGnQINPqFQdzIFrEItGQbQYh9oD83m9Xc5SOjLYzlY8vNyAMcftyNeF@aliyun-proxy.pjlab.org.cn:13128"
# export https_proxy="https://zhoujiecheng:SxtTJSGnQINPqFQdzIFrEItGQbQYh9oD83m9Xc5SOjLYzlY8vNyAMcftyNeF@aliyun-proxy.pjlab.org.cn:13128"
# export HTTPS_PROXY="https://zhoujiecheng:SxtTJSGnQINPqFQdzIFrEItGQbQYh9oD83m9Xc5SOjLYzlY8vNyAMcftyNeF@aliyun-proxy.pjlab.org.cn:13128"
# export HTTP_PROXY="https://zhoujiecheng:SxtTJSGnQINPqFQdzIFrEItGQbQYh9oD83m9Xc5SOjLYzlY8vNyAMcftyNeF@aliyun-proxy.pjlab.org.cn:13128"

wandb login cac422113ce38ff90b619824d9cb1277b557f082
export WANDB_API_KEY=cac422113ce38ff90b619824d9cb1277b557f082

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/cpfs01/user/zhoujiecheng/workload_rl_analyse/models/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-gsm8k-async-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl/data/gsm8k_multi_turn/train.parquet \
    data.val_files=/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl/data/gsm8k_multi_turn/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=10 $@

