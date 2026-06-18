# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import json
import os
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


from rllm.rewards.rl_reward import rllm_reward_fn

def _replay_reward_time_s(workload_path, step):
    reward_file = os.getenv("WORKLOAD_REWARD_TIMINGS_FILE")
    if reward_file is None:
        reward_file = os.path.join(workload_path, f"reward_timings_step_{step}.jsonl")
    if not os.path.exists(reward_file):
        return 0.0

    reward_times = []
    with open(reward_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {reward_file}:{line_no}: {exc}") from exc
            if record.get("validate", False):
                continue
            if str(record.get("step", step)) != str(step):
                continue
            reward_times.append(float(record.get("reward_time_s", 0.0)))
    return max(reward_times) if reward_times else 0.0


def _add_replayed_reward_time(timing_s, metrics, workload_path, step, num_samples):
    reward_time_s = _replay_reward_time_s(workload_path, step)
    if reward_time_s <= 0:
        return timing_s, metrics

    timing_s = dict(timing_s)
    metrics = dict(metrics)
    sleep_reward = os.getenv("WORKLOAD_BENCHMARK_SLEEP_REWARD", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    gen_observed = float(timing_s.get("gen", 0.0))
    step_observed = float(timing_s.get("step", 0.0))
    timing_s["rollout_reward_replay"] = reward_time_s

    if sleep_reward:
        gen_model_only = max(0.0, gen_observed - reward_time_s)
        step_model_only = max(0.0, step_observed - reward_time_s)
        timing_s["gen_model_only_est"] = gen_model_only
        timing_s["step_model_only_est"] = step_model_only
    else:
        gen_model_only = gen_observed
        step_model_only = step_observed
        timing_s["gen_model_only"] = gen_model_only
        timing_s["gen"] = gen_model_only + reward_time_s
        timing_s["step_model_only"] = step_model_only
        timing_s["step"] = step_model_only + reward_time_s

    metrics["timing_s/rollout_reward_replay"] = reward_time_s
    if sleep_reward:
        metrics["timing_s/gen_model_only_est"] = gen_model_only
        metrics["timing_s/step_model_only_est"] = step_model_only
    else:
        metrics["timing_s/gen_model_only"] = gen_model_only
        metrics["timing_s/gen"] = timing_s["gen"]
        metrics["timing_s/step_model_only"] = step_model_only
        metrics["timing_s/step"] = timing_s["step"]

    response_tokens = float(metrics.get("response_length/mean", 0.0)) * max(int(num_samples), 1)
    if response_tokens > 0:
        key_suffix = "gen_model_only_est" if sleep_reward else "gen_model_only"
        metrics[f"timing_per_token_ms/{key_suffix}"] = gen_model_only * 1000.0 / response_tokens
        metrics["timing_per_token_ms/rollout_reward_replay"] = reward_time_s * 1000.0 / response_tokens
        metrics["timing_per_token_ms/gen"] = float(timing_s.get("gen", gen_observed)) * 1000.0 / response_tokens
    return timing_s, metrics


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return rllm_reward_fn


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(data_source=data_source, llm_solution=sequences_str, ground_truth=ground_truth)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=48) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'WANDB_API_KEY':'cac422113ce38ff90b619824d9cb1277b557f082'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    if os.getenv("WORKLOAD_BENCHMARK", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
        from verl.trainer.Generator import Generator

        default_workload_path = os.path.join(
            os.getenv("RAY_DATA_HOME", "/rllm"),
            "profile/packed_length_log",
        )
        workload_path = os.getenv("WORKLOAD_PATH") or os.getenv("WORKLOAD_LENGTH_DIR") or default_workload_path
        workload_step_groups_env = os.getenv("WORKLOAD_STEP_GROUPS", "0")
        workload_steps = []
        for group in workload_step_groups_env.split(";"):
            group = group.strip()
            if not group:
                continue
            step_items = [item.strip() for item in group.split(",") if item.strip()]
            if len(step_items) != 1:
                raise ValueError(
                    "rllm benchmark does not support cross-step sampling; "
                    f"each WORKLOAD_STEP_GROUPS group must contain one step, got {group!r}"
                )
            workload_steps.append(step_items[0])
        if not workload_steps:
            workload_steps = ["0"]

        print(f"Benchmark workload path: {workload_path}, step groups: {workload_steps}")
        generator = Generator(
            workload_path,
            local_path,
            max_prompt_length=int(config.data.max_prompt_length),
        )
        num_prompts = int(config.data.train_batch_size)
        n_samples = int(config.actor_rollout_ref.rollout.n)
        benchmark_repeat = int(os.getenv("WORKLOAD_BENCHMARK_REPEAT", "1"))
        for repeat_idx in range(benchmark_repeat):
            print(f"Benchmark repeat {repeat_idx + 1}/{benchmark_repeat}")
            for group_idx, step in enumerate(workload_steps):
                print(
                    f"Benchmark repeat {repeat_idx + 1}/{benchmark_repeat}, "
                    f"group {group_idx + 1}/{len(workload_steps)}, step: {step}"
                )
                benchmark_input = generator.generate(bsz=num_prompts, n_samples=n_samples, step=step)
                timing_s, metrics = trainer.benchmark_fit(benchmark_input)
                timing_s, metrics = _add_replayed_reward_time(
                    timing_s,
                    metrics,
                    workload_path=workload_path,
                    step=step,
                    num_samples=num_prompts * n_samples,
                )
                print(f"benchmark_repeat/{repeat_idx + 1}/group/{group_idx + 1}/step/{step}/timing_s: {timing_s}")
                print(f"benchmark_repeat/{repeat_idx + 1}/group/{group_idx + 1}/step/{step}/metrics: {metrics}")
    else:
        trainer.fit()


if __name__ == '__main__':
    main()
