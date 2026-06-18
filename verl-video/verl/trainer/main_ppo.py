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


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars":
                {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "WANDB_API_KEY":"cac422113ce38ff90b619824d9cb1277b557f082",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config, num_gpus=8))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def __init__(self):
        self.global_pool_id = "global_pool_id"

    def _get_worker_classes(self, config):
        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError
        return actor_rollout_cls, ray_worker_group_cls, CriticWorker

    def _get_reward_worker_cls(self, config):
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        return RewardModelWorker

    def _get_tokenizer(self, config):
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        return tokenizer

    def _get_processor(self, config):
        from verl.utils import hf_processor
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        processor = hf_processor(local_path, use_fast=True)
        return processor

    def _get_dataset(self, config, tokenizer, processor):
        from verl.utils.dataset import RLHFDataset
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = RLHFDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = RLHFDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        return train_dataset, val_dataset, collate_fn

    def run(self, config, num_gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.ppo.reward import load_reward_manager
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # Core components
        tokenizer, processor = self._get_tokenizer(config), self._get_processor(config)
        actor_rollout_cls, ray_worker_group_cls, critic_worker_cls = self._get_worker_classes(config)

        ROLE_WORKER_MAPPING = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(critic_worker_cls),
        }

        self.resource_pool_spec = {
            self.global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        mapping = {
            Role.ActorRollout: self.global_pool_id,
            Role.Critic: self.global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            ROLE_WORKER_MAPPING[Role.RewardModel] = ray.remote(self._get_reward_worker_cls(config))
            mapping[Role.RewardModel] = self.global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            ROLE_WORKER_MAPPING[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = self.global_pool_id

        # Core components
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=self.resource_pool_spec, mapping=mapping)

        train_dataset, val_dataset, collate_fn = self._get_dataset(config, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=ROLE_WORKER_MAPPING,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        if os.getenv("WORKLOAD_BENCHMARK", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            from .Generator import Generator

            default_workload_path = "/verl-video/logs/packed_length_log/20260610_053922"
            workload_step = os.getenv("WORKLOAD_STEP", "1")
            workload_path = os.getenv("WORKLOAD_PATH") or os.getenv("OUTPUT_LEN_FILE")
            workload_length_dir = os.getenv("WORKLOAD_LENGTH_DIR")
            if workload_path is None:
                packed_step_path = None
                if workload_length_dir:
                    packed_step_path = os.path.join(workload_length_dir, f"packed_lengths_step_{workload_step}.jsonl")
                workload_path = workload_length_dir if packed_step_path and os.path.exists(packed_step_path) else default_workload_path
            print(f"Benchmark workload path: {workload_path}, step: {workload_step}")
            model_path = os.path.expanduser(config.actor_rollout_ref.model.path)
            generator = Generator(workload_path, model_path)
            num_prompts = int(config.data.train_batch_size)
            n_samples = int(config.actor_rollout_ref.rollout.n)
            benchmark_repeat = int(os.getenv("WORKLOAD_BENCHMARK_REPEAT", "1"))
            for repeat_idx in range(benchmark_repeat):
                print(f"Benchmark repeat {repeat_idx + 1}/{benchmark_repeat}")
                input = generator.generate(bsz=num_prompts, n_samples=n_samples, step=workload_step)
                timing_s, metrics = trainer.benchmark_fit(input)
                print(f"benchmark_repeat/{repeat_idx + 1}/timing_s: {timing_s}")
                print(f"benchmark_repeat/{repeat_idx + 1}/metrics: {metrics}")
        else:
            trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
