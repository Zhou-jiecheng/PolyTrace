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

import os

import hydra
import ray

from .dapo_ray_trainer import RayDAPOTrainer


def get_custom_reward_fn(config):
    import importlib.util

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
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
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        if os.getenv("WORKLOAD_BENCHMARK", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            from verl.trainer.Generator import Generator

            default_workload_path = os.path.join(
                os.getenv("RAY_DATA_HOME", "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl"),
                "profile/packed_length_log",
            )
            workload_step = os.getenv("WORKLOAD_STEP", "0")
            workload_steps_env = os.getenv("WORKLOAD_STEPS", workload_step)
            workload_step_groups_env = os.getenv("WORKLOAD_STEP_GROUPS")
            if not workload_step_groups_env and ";" in workload_steps_env:
                workload_step_groups_env = workload_steps_env
            workload_steps = [
                item.strip()
                for item in workload_steps_env.split(",")
                if item.strip()
            ]
            workload_step_groups = None
            if workload_step_groups_env:
                workload_step_groups = [
                    [item.strip() for item in group.split(",") if item.strip()]
                    for group in workload_step_groups_env.split(";")
                    if group.strip()
                ]
                workload_step_groups = [group for group in workload_step_groups if group]
                if workload_step_groups:
                    workload_steps = workload_step_groups[0]
            workload_path = os.getenv("WORKLOAD_PATH")
            workload_length_dir = os.getenv("WORKLOAD_LENGTH_DIR")
            if workload_path is None:
                packed_step_path = None
                if workload_length_dir:
                    packed_step_path = os.path.join(workload_length_dir, f"packed_lengths_step_{workload_steps[0]}.jsonl")
                workload_path = workload_length_dir if packed_step_path and os.path.exists(packed_step_path) else default_workload_path
            if not os.getenv("WORKLOAD_FILTER_INDICES_FILE") and os.path.isdir(workload_path):
                filter_indices_file = os.path.join(workload_path, "dapo_filter_indices.jsonl")
                if os.path.exists(filter_indices_file):
                    os.environ["WORKLOAD_FILTER_INDICES_FILE"] = filter_indices_file
            print(f"Benchmark workload path: {workload_path}, steps: {workload_steps}")
            generator = Generator(
                workload_path,
                local_path,
                max_prompt_length=int(config.data.max_prompt_length),
            )
            num_prompts = int(config.data.gen_batch_size)
            n_samples = int(config.actor_rollout_ref.rollout.n)
            benchmark_repeat = int(os.getenv("WORKLOAD_BENCHMARK_REPEAT", "1"))
            if workload_step_groups is None:
                workload_step_groups = [workload_steps]
            for repeat_idx in range(benchmark_repeat):
                print(f"Benchmark repeat {repeat_idx + 1}/{benchmark_repeat}")
                for group_idx, step_group in enumerate(workload_step_groups):
                    print(
                        f"Benchmark repeat {repeat_idx + 1}/{benchmark_repeat}, "
                        f"group {group_idx + 1}/{len(workload_step_groups)}, steps: {step_group}"
                    )
                    inputs = [
                        generator.generate(bsz=num_prompts, n_samples=n_samples, step=step)
                        for step in step_group
                    ]
                    timing_s, metrics = trainer.benchmark_fit(inputs)
                    step_group_name = ",".join(step_group)
                    print(f"benchmark_repeat/{repeat_idx + 1}/group/{group_idx + 1}/steps/{step_group_name}/timing_s: {timing_s}")
                    print(f"benchmark_repeat/{repeat_idx + 1}/group/{group_idx + 1}/steps/{step_group_name}/metrics: {metrics}")
        else:
            trainer.fit()


if __name__ == "__main__":
    main()
