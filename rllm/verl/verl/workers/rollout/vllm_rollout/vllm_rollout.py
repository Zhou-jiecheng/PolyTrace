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
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from copy import deepcopy
import json
import os
import time
from datetime import datetime

import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
import traceback
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _workload_env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _workload_default_dir() -> str:
    run_id = os.getenv("WORKLOAD_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_home = os.getenv("RAY_DATA_HOME", "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/rllm")
    return os.path.join(data_home, "profile", "packed_length_log", run_id)


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
        alpha = min(1.0, max(0.0, float(os.getenv("WORKLOAD_MAX_TOKEN_ALPHA", "0.5"))))
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


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, reward_fn, val_reward_fn, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.actor_module = actor_module
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        self.tensor_parallel_size = tensor_parallel_size
        
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 32768)
        
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(actor_module,
                                    tokenizer=tokenizer,
                                    model_hf_config=model_hf_config,
                                    tensor_parallel_size=tensor_parallel_size,
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=config.prompt_length + config.response_length,
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    enable_chunked_prefill=config.enable_chunked_prefill,
                                    load_format=config.load_format)
        self.tensor_parallel_rank = vllm_ps.get_tensor_model_parallel_rank()
        print(f"Rank {torch.distributed.get_rank()}, TP rank {self.tensor_parallel_rank} initialized vLLM rollout")
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self._workload_step = 0

    def _write_workload_jsonl(self, filename: str, records: list[dict]):
        if not records:
            return
        workload_dir = os.getenv("WORKLOAD_LENGTH_DIR") or _workload_default_dir()
        os.makedirs(workload_dir, exist_ok=True)
        path = os.path.join(workload_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _collect_workload_lengths(
        self,
        *,
        input_seqlen_lst: list[int],
        output_seqlen_lst: list[int],
        n_samples: int,
        is_validation: bool,
    ):
        if not _workload_env_flag("WORKLOAD_COLLECT_LENGTHS"):
            return
        if is_validation and not _workload_env_flag("WORKLOAD_COLLECT_VALIDATE_LENGTHS"):
            return
        if self.tensor_parallel_rank != 0:
            return

        step = self._workload_step
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dp_rank = rank // max(int(self.tensor_parallel_size), 1)
        records = []
        for local_index, input_len in enumerate(input_seqlen_lst):
            start = local_index * n_samples
            end = start + n_samples
            output_group = output_seqlen_lst[start:end]
            if len(output_group) != n_samples:
                continue
            records.append({
                "step": step,
                "rank": rank,
                "tp_rank": int(self.tensor_parallel_rank),
                "dp_rank": dp_rank,
                "local_index": local_index,
                "input": int(input_len),
                "output": [int(x) for x in output_group],
                "n_sampling": int(n_samples),
                "validate": bool(is_validation),
            })
        self._write_workload_jsonl(f"packed_lengths_step_{step}.jsonl", records)

    def _collect_reward_timing(
        self,
        *,
        reward_time_s: float,
        batch_size: int,
        response_seqlen_lst: list[int],
        is_validation: bool,
    ):
        if not _workload_env_flag("WORKLOAD_COLLECT_REWARD_TIMINGS", os.getenv("WORKLOAD_COLLECT_LENGTHS", "0")):
            return
        if is_validation and not _workload_env_flag("WORKLOAD_COLLECT_VALIDATE_LENGTHS"):
            return
        if self.tensor_parallel_rank != 0:
            return

        step = self._workload_step
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dp_rank = rank // max(int(self.tensor_parallel_size), 1)
        record = {
            "step": step,
            "rank": rank,
            "tp_rank": int(self.tensor_parallel_rank),
            "dp_rank": dp_rank,
            "batch_size": int(batch_size),
            "reward_time_s": float(reward_time_s),
            "response_sum": int(sum(response_seqlen_lst)),
            "response_mean": float(np.mean(response_seqlen_lst)) if response_seqlen_lst else 0.0,
            "response_max": int(max(response_seqlen_lst)) if response_seqlen_lst else 0,
            "validate": bool(is_validation),
        }
        self._write_workload_jsonl(f"reward_timings_step_{step}.jsonl", [record])

    def _sampling_params_for_workload(self, max_tokens):
        kwargs = dict(
            n=1,
            logprobs=1,
            max_tokens=max(1, min(int(self.config.response_length), int(max_tokens))),
            top_p=1,
            top_k=-1,
            temperature=1.0,
            ignore_eos=True,
        )
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False
        return SamplingParams(**kwargs)

    def _enforce_response_token_length(self, response, target_lengths, eos_token_id):
        if target_lengths is None:
            return response
        response = response.clone()
        max_len = response.size(1)
        filler_token_id = self.pad_token_id if self.pad_token_id != eos_token_id else 0
        for row_idx, target_len in enumerate(target_lengths):
            target_len = max(1, min(max_len, int(target_len)))
            row = response[row_idx]
            if target_len > 1:
                early_eos = row[: target_len - 1].eq(eos_token_id)
                row[: target_len - 1][early_eos] = filler_token_id
            row[target_len - 1] = eos_token_id
            if target_len < max_len:
                row[target_len:] = self.pad_token_id
        return response

    def _dummy_reward_tensor(self, response, response_lengths):
        reward_tensor = torch.zeros_like(response, dtype=torch.float32)
        for idx, length in enumerate(response_lengths):
            pos = max(0, min(int(length), response.size(1)) - 1)
            reward_tensor[idx, pos] = float(idx % 2)
        return reward_tensor

    def _replay_reward_sleep_time_s(self, is_validation):
        if is_validation:
            return 0.0
        if not _workload_env_flag("WORKLOAD_BENCHMARK_SLEEP_REWARD"):
            return 0.0
        if self.tensor_parallel_rank != 0:
            return 0.0

        step = self._workload_step
        workload_path = os.getenv("WORKLOAD_PATH") or os.getenv("WORKLOAD_LENGTH_DIR") or _workload_default_dir()
        reward_file = os.getenv("WORKLOAD_REWARD_TIMINGS_FILE")
        if reward_file is None:
            reward_file = os.path.join(workload_path, f"reward_timings_step_{step}.jsonl")
        if not os.path.exists(reward_file):
            return 0.0

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dp_rank = rank // max(int(self.tensor_parallel_size), 1)
        fallback_sleep_s = 0.0
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
                reward_time_s = float(record.get("reward_time_s", 0.0))
                fallback_sleep_s = max(fallback_sleep_s, reward_time_s)
                if int(record.get("dp_rank", -1)) == int(dp_rank):
                    return max(0.0, reward_time_s)
        return max(0.0, fallback_sleep_s)

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

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto,  **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        # Rebuild vLLM cache engine if configured
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        # Extract input tensors from prompt batch
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)
        
        # Pre-process input token ids
        idx_list = [
            _pre_process_inputs(self.pad_token_id, idx[i])
            for i in range(batch_size)
        ]
        input_seqlen_lst = [int(x) for x in attention_mask.view(batch_size, -1).sum(-1).tolist()]
        # Configure sampling parameters
        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 0.95,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1
            }
        is_validation = False
        if prompts.meta_info.get('val_temperature', None):
            kwargs['temperature'] = prompts.meta_info['val_temperature']
            is_validation = True

        kwargs['n'] = 1
        n_samples = int(self.config.n) if do_sample else 1
        non_tensor_batch = deepcopy(prompts.non_tensor_batch)
        workload_output_groups = None
        workload_target_lengths = None
        workload_sampling_params = None
        if "output_len" in non_tensor_batch:
            workload_output_groups = _normalize_output_length_groups(
                non_tensor_batch.pop("output_len"),
                rollout_n=n_samples,
                batch_size=batch_size,
            )

        if do_sample:
            idx_list = [deepcopy(item) for item in idx_list for _ in range(n_samples)]

        if workload_output_groups is not None:
            workload_policy = os.getenv("WORKLOAD_MAX_TOKEN_POLICY", "adaptive_blend").strip().lower()
            workload_separate_samples = workload_policy in {"separate", "split", "split_samples", "exact"}
            workload_target_lengths = []
            workload_sampling_params = []
            for output_group in workload_output_groups:
                estimated_max_tokens = None if workload_separate_samples else _estimate_workload_max_tokens(output_group)
                for target_len in output_group:
                    target_len = max(1, min(int(self.config.response_length), int(target_len)))
                    workload_target_lengths.append(target_len)
                    max_tokens = target_len if workload_separate_samples else estimated_max_tokens
                    workload_sampling_params.append(self._sampling_params_for_workload(max_tokens))

        sampling_params = workload_sampling_params if workload_sampling_params is not None else self.sampling_params

        # Generate sequences
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # Process outputs
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        # Pad sequences if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(
                response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(
                log_probs, self.config.response_length, self.pad_token_id)
        response = self._enforce_response_token_length(response, workload_target_lengths, eos_token_id)

        # Handle multiple samples per prompt
        if n_samples > 1 and do_sample:
            idx = idx.repeat_interleave(n_samples, dim=0)
            attention_mask = attention_mask.repeat_interleave(
                n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(
                n_samples, dim=0)
            batch_size = batch_size * n_samples
                # Create interleaved non_tensor_batch
            prompt_non_tensor_batch = non_tensor_batch
            non_tensor_batch = {}
            for key, val in prompt_non_tensor_batch.items():
                # Repeat each element n times (interleaved)
                repeated_val = np.repeat(val, n_samples)
                non_tensor_batch[key] = repeated_val

        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids],
                                dim=-1)
        response_attention_mask = get_eos_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype)
        attention_mask = torch.cat(
            (attention_mask, response_attention_mask), dim=-1)
        output_seqlen_lst = [int(x) for x in response_attention_mask.view(batch_size, -1).sum(-1).tolist()]
        self._collect_workload_lengths(
            input_seqlen_lst=input_seqlen_lst,
            output_seqlen_lst=output_seqlen_lst,
            n_samples=n_samples,
            is_validation=is_validation,
        )
        # Construct output batch
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # Free cache if configured
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        batch = DataProto(batch=batch,
                            non_tensor_batch=non_tensor_batch,
                            meta_info=prompts.meta_info)
        if self.reward_fn is not None and not is_validation:
            if _workload_env_flag("WORKLOAD_BENCHMARK_SKIP_REWARD") and workload_output_groups is not None:
                reward_sleep_s = self._replay_reward_sleep_time_s(is_validation=is_validation)
                if reward_sleep_s > 0:
                    time.sleep(reward_sleep_s)
                reward_tensor = self._dummy_reward_tensor(response, output_seqlen_lst)
                batch.batch['token_level_scores'] = reward_tensor
            else:
                reward_start = time.perf_counter()
                reward_tensor = self.reward_fn(batch)
                reward_time_s = time.perf_counter() - reward_start
                batch.batch['token_level_scores'] = reward_tensor
                self._collect_reward_timing(
                    reward_time_s=reward_time_s,
                    batch_size=batch_size,
                    response_seqlen_lst=output_seqlen_lst,
                    is_validation=is_validation,
                )

        if (not is_validation) or _workload_env_flag("WORKLOAD_COLLECT_VALIDATE_LENGTHS"):
            self._workload_step += 1
        return batch
