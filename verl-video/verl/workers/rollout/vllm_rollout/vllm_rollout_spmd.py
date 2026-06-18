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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import fcntl
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Union
import json
import random

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int_list(value):
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return [int(item) for item in values]


def _normalize_output_length_groups(output_lens, rollout_n, batch_size):
    if len(output_lens) != batch_size:
        raise AssertionError(
            f"output_len first dimension {len(output_lens)} should equal prompt batch size {batch_size}"
        )

    output_groups = []
    for prompt_idx, output_len in enumerate(output_lens):
        values = _as_int_list(output_len)
        if len(values) == rollout_n:
            output_groups.append(values)
        elif len(values) == 1:
            output_groups.append(values * rollout_n)
        elif rollout_n == 1:
            output_groups.append([values[0]])
        else:
            raise AssertionError(
                f"output_len[{prompt_idx}] has {len(values)} values, expected 1 or rollout n={rollout_n}"
            )

    return output_groups


def _normalize_output_lengths(output_lens, rollout_n, batch_size):
    return [
        value
        for group in _normalize_output_length_groups(output_lens, rollout_n, batch_size)
        for value in group
    ]


def _estimate_workload_max_tokens(output_group):
    values = [max(1, int(value)) for value in output_group]
    policy = os.getenv("WORKLOAD_MAX_TOKEN_POLICY", "adaptive_blend").strip().lower()
    if policy == "max":
        estimate = max(values)
    elif policy == "mean":
        estimate = int(np.ceil(float(np.mean(values))))
    elif policy == "p75":
        estimate = int(np.ceil(float(np.percentile(values, 75))))
    elif policy == "p875":
        estimate = int(np.ceil(float(np.percentile(values, 87.5))))
    elif policy == "blend":
        alpha = float(os.getenv("WORKLOAD_MAX_TOKEN_ALPHA", "0.5"))
        alpha = min(1.0, max(0.0, alpha))
        mean_value = float(np.mean(values))
        estimate = int(np.ceil(mean_value + alpha * (max(values) - mean_value)))
    elif policy in {"adaptive_blend", "auto_blend"}:
        arr = np.asarray(values, dtype=np.float64)
        mean_value = float(np.mean(arr))
        max_value = float(np.max(arr))
        p75_value = float(np.percentile(arr, 75))
        if max_value <= mean_value:
            alpha = 0.0
        else:
            min_value = float(np.min(arr))
            tail_ratio_full = max(1.0 + 1e-6, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_RATIO_FULL", "3.3")))
            tail_cv_soft = max(1e-6, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_CV_SOFT", "1.0")))
            tail_power = max(1e-6, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_POWER", "1.0")))
            ratio_weight = max(0.0, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_RATIO_WEIGHT", "0.7")))
            cv_weight = max(0.0, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_CV_WEIGHT", "0.2")))
            gap_weight = max(0.0, float(os.getenv("WORKLOAD_ADAPTIVE_TAIL_GAP_WEIGHT", "0.1")))

            mean_safe = max(mean_value, 1e-6)
            ratio_score = np.log(max(max_value / mean_safe, 1.0)) / np.log(tail_ratio_full)
            cv = float(np.std(arr) / mean_safe)
            cv_score = cv / (cv + tail_cv_soft)
            gap_score = 0.0 if max_value <= min_value else (max_value - p75_value) / max(max_value - min_value, 1e-6)
            weight_sum = max(ratio_weight + cv_weight + gap_weight, 1e-6)
            tail_score = (ratio_weight * ratio_score + cv_weight * cv_score + gap_weight * gap_score) / weight_sum
            tail_score = min(1.0, max(0.0, tail_score)) ** tail_power

            alpha = tail_score * float(os.getenv("WORKLOAD_MAX_TOKEN_ALPHA_SCALE", "1.0"))
        alpha_cap = min(1.0, max(0.0, float(os.getenv("WORKLOAD_MAX_TOKEN_ALPHA_CAP", "1.0"))))
        alpha_floor = min(1.0, max(0.0, float(os.getenv("WORKLOAD_MAX_TOKEN_ALPHA_FLOOR", "0.0"))))
        alpha = min(alpha_cap, max(alpha_floor, alpha))
        estimate = int(np.ceil(mean_value + alpha * (max_value - mean_value)))
    elif policy == "mean_margin":
        margin = float(os.getenv("WORKLOAD_MAX_TOKEN_MARGIN", "1.05"))
        estimate = int(np.ceil(float(np.mean(values)) * margin))
    else:
        raise ValueError(f"Unknown WORKLOAD_MAX_TOKEN_POLICY={policy!r}")
    return max(1, min(max(values), estimate))


def _append_jsonl(path, records):
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics
def get_length_info(file_path, step):
    """get specific step's input/output length from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if step in data:
        input_len = data[step].get('input', [])
        output_len = data[step].get('output_groups', data[step].get('output', []))
        return input_len, output_len
    else:
        print(f"Step {step} not found in the data.")
        return None, None

def sample_from_range_distribution(original_list, target_count):
    """
    sample target count numbers from a range distribution based on the original list.
    """
    if not original_list:
        return []
    
    # if original list is less than target count, return a shuffled copy of the original list
    if target_count >= len(original_list):
        result = original_list.copy()
        random.shuffle(result)
        return result
    
    min_val = min(original_list)
    max_val = max(original_list)
    mean_val = sum(original_list) / len(original_list)
    
    # count bins based on the range of values
    num_bins = min(20, len(set(original_list)))  # at most 20 bins
    bin_size = (max_val - min_val) / num_bins if num_bins > 1 else 1
    
    if bin_size == 0: 
        return [original_list[0]] * target_count
    
    bin_counts = [0] * num_bins
    for val in original_list:
        bin_idx = min(int((val - min_val) / bin_size), num_bins - 1)
        bin_counts[bin_idx] += 1
    
    result = []
    for i in range(num_bins):
        if bin_counts[i] == 0:
            continue
            
        proportion = bin_counts[i] / len(original_list)
        samples_needed = int(proportion * target_count)
        
        if samples_needed > 0:
            bin_start = min_val + i * bin_size
            bin_end = min_val + (i + 1) * bin_size
            
            values_in_bin = [v for v in original_list if bin_start <= v < bin_end or (i == num_bins - 1 and v == max_val)]
            
            if values_in_bin:
                samples_in_bin = random.choices(values_in_bin, k=min(samples_needed, len(values_in_bin)))
                result.extend(samples_in_bin)

    while len(result) < target_count:
        result.append(random.choice(original_list))
    
    if len(result) > target_count:
        result = random.sample(result, target_count)
    
    random.shuffle(result)
    return result



# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, 'rope_scaling', None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)
        self.step = 0

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_kwargs = kwargs
        self.sampling_params = SamplingParams(**kwargs)
        self.benchmark = _env_flag("WORKLOAD_BENCHMARK")
        self.output_length_path = os.getenv("OUTPUT_LEN_FILE", "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/logs/cosmos_workloads.json")
        self.collect_workload_lengths = _env_flag("WORKLOAD_COLLECT_LENGTHS")
        self.collect_validate_workload_lengths = _env_flag("WORKLOAD_COLLECT_VALIDATE_LENGTHS")
        self.workload_length_dir = os.getenv("WORKLOAD_LENGTH_DIR", "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/verl-video/logs/packed_length_log")
        self.workload_length_file = os.getenv("WORKLOAD_LENGTH_FILE", "")

        self.pad_token_id = tokenizer.pad_token_id

    def _enforce_response_token_length(self, token_ids, target_len, eos_token_id):
        target_len = min(int(target_len), int(self.config.response_length))
        if target_len <= 0:
            return []

        eos_token_ids = _as_int_list(eos_token_id)
        eos_token_id = int(eos_token_ids[0]) if eos_token_ids else int(self.pad_token_id)
        token_ids = [int(token_id) for token_id in token_ids[:target_len]]
        if len(token_ids) < target_len:
            token_ids.extend([int(self.pad_token_id)] * (target_len - len(token_ids)))
        token_ids[-1] = eos_token_id
        return token_ids

    def _tp_rank(self):
        try:
            return int(vllm_ps.get_tensor_model_parallel_rank())
        except Exception:
            if torch.distributed.is_initialized():
                tp_size = int(self.config.get("tensor_model_parallel_size", 1))
                return int(torch.distributed.get_rank() % tp_size)
            return 0

    def _tp_size(self):
        try:
            return int(vllm_ps.get_tensor_model_parallel_world_size())
        except Exception:
            return int(self.config.get("tensor_model_parallel_size", 1))

    def _should_dump_workload_lengths(self, is_validate):
        if not self.collect_workload_lengths:
            return False
        if is_validate and not self.collect_validate_workload_lengths:
            return False
        return self._tp_rank() == 0

    def _workload_length_path(self, step):
        if self.workload_length_file:
            return self.workload_length_file
        return os.path.join(self.workload_length_dir, f"packed_lengths_step_{step}.jsonl")

    def _dump_workload_lengths(self, prompt_lengths, output_length_groups, is_validate):
        if not self._should_dump_workload_lengths(is_validate):
            return

        rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
        tp_size = self._tp_size()
        records = []
        for local_index, (input_len, output_lens) in enumerate(zip(prompt_lengths, output_length_groups)):
            records.append(
                {
                    "step": int(self.step),
                    "rank": rank,
                    "dp_rank": rank // tp_size if tp_size > 0 else rank,
                    "tp_rank": self._tp_rank(),
                    "local_index": int(local_index),
                    "input": int(input_len),
                    "output": [int(output_len) for output_len in output_lens],
                    "n_sampling": int(len(output_lens)),
                    "validate": bool(is_validate),
                }
            )
        _append_jsonl(self._workload_length_path(self.step), records)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        prompt_batch_size = batch_size
        prompt_lengths = attention_mask.view(prompt_batch_size, -1).sum(-1).tolist()

        # input_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()
        # for input_seq_len in input_seqlen_lst:
        #     with open(f"/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/verl-video/logs/GEO-3K/step_input_length_{self.step}.log", "a") as f:
        #         f.write(f"step{self.step}, input: {input_seq_len}\n")
        # if self.benchmark:
        #     step_input, step_output = get_length_info(self.output_length_path, os.getenv("WORKLOAD_STEP", "0"))
        #     assert step_output, f"step_output is None, please check the output length file {self.output_length_path}"
        #     # sample_step_output =  prompts.non_tensor_batch.pop('output_len')
        #     sample_step_output = sample_from_range_distribution(step_output, batch_size * self.sampling_kwargs["n"])
        
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        use_workload_output_len = "output_len" in non_tensor_batch
        workload_rollout_n = 1
        workload_output_groups = None
        workload_target_lengths = None
        workload_separate_samples = False
        sampling_params = self.sampling_params
        if use_workload_output_len:
            # One prompt owns n sampled output lengths in the replay workload.
            workload_rollout_n = int(self.sampling_kwargs["n"] if do_sample and not is_validate else 1)
            workload_output_groups = _normalize_output_length_groups(
                non_tensor_batch.pop("output_len"),
                rollout_n=workload_rollout_n,
                batch_size=batch_size,
            )
            workload_policy = os.getenv("WORKLOAD_MAX_TOKEN_POLICY", "adaptive_blend").strip().lower()
            workload_separate_samples = workload_policy in {"separate", "split", "split_samples", "exact"}
            sampling_params_list = []
            if workload_separate_samples:
                # Baseline mode: replay 64*8 as 512*1 requests. This precisely
                # controls each sample length, but intentionally pays repeated
                # prefill cost because the prompt is not shared across n samples.
                expanded_vllm_inputs = []
                workload_target_lengths = []
                for input_data, output_group in zip(vllm_inputs, workload_output_groups):
                    for target_len in output_group:
                        expanded_input = dict(input_data)
                        if isinstance(expanded_input.get("prompt_token_ids"), list):
                            expanded_input["prompt_token_ids"] = list(expanded_input["prompt_token_ids"])
                        expanded_vllm_inputs.append(expanded_input)
                        target_len = max(1, min(int(self.config.response_length), int(target_len)))
                        workload_target_lengths.append(target_len)
                        sampling_params_list.append(
                            SamplingParams(
                                top_p=1,
                                top_k=-1,
                                temperature=1.0,
                                n=1,
                                ignore_eos=True,
                                max_tokens=target_len,
                            )
                        )
                vllm_inputs = expanded_vllm_inputs
            else:
                # vLLM accepts one max_tokens per request. Replaying with the
                # group max overestimates decode work when one sample is a long
                # outlier, so use a calibrated estimate while keeping n samples
                # in one request to preserve the real prompt-sharing shape.
                for output_group in workload_output_groups:
                    sampling_kwargs = {
                        "top_p": 1,
                        "top_k": -1,
                        "temperature": 1.0,
                        "n": workload_rollout_n,
                        "ignore_eos": True,
                        "max_tokens": min(
                            int(self.config.response_length),
                            _estimate_workload_max_tokens(output_group),
                        ),
                    }
                    sampling_params_list.append(SamplingParams(**sampling_kwargs))
            sampling_params = sampling_params_list
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            response = []
            for output_idx, output in enumerate(outputs):
                for sample_id in range(len(output.outputs)):
                    token_ids = [int(token_id) for token_id in output.outputs[sample_id].token_ids]
                    if workload_output_groups is not None:
                        if workload_separate_samples:
                            target_len = workload_target_lengths[len(response)]
                        else:
                            target_len = workload_output_groups[output_idx][sample_id]
                        token_ids = self._enforce_response_token_length(
                            token_ids,
                            target_len,
                            eos_token_id,
                        )
                    response.append(token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

            if use_workload_output_len or (self.benchmark and self.sampling_kwargs["n"] > 1 and do_sample):
                repeat_times = workload_rollout_n if use_workload_output_len else self.sampling_kwargs["n"]
                idx = _repeat_interleave(idx, repeat_times)
                attention_mask = _repeat_interleave(attention_mask, repeat_times)
                position_ids = _repeat_interleave(position_ids, repeat_times)
                batch_size = batch_size * repeat_times
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], repeat_times)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], repeat_times)
            elif not self.benchmark and self.sampling_kwargs["n"] > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)        

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        if self.collect_workload_lengths:
            output_lengths = response_attention_mask.view(batch_size, -1).sum(-1).tolist()
            outputs_per_prompt = batch_size // prompt_batch_size if prompt_batch_size > 0 else 0
            output_length_groups = [
                output_lengths[i * outputs_per_prompt : (i + 1) * outputs_per_prompt]
                for i in range(prompt_batch_size)
            ]
            self._dump_workload_lengths(prompt_lengths, output_length_groups, is_validate=is_validate)

        # output_seqlen_lst = response_attention_mask.view(batch_size, -1).sum(-1).tolist()
        # for input_seq_len in output_seqlen_lst:
        #     with open(f"/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/verl-video/logs/GEO-3K/output_length_{self.step}.log", "a") as f:
        #         f.write(f"step{self.step}, input: {input_seq_len}\n")
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        self.step += 1
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
