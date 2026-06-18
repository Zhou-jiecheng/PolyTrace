# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from copy import deepcopy
from json import JSONDecodeError
from typing import TYPE_CHECKING, Union
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_ip, get_open_port
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.utils.debug import GPUMemoryLogger
from verl.utils.model import compute_position_id_with_mask
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs, _pre_process_inputs
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _workload_env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _workload_default_dir() -> str:
    run_id = os.getenv("WORKLOAD_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_home = os.getenv("RAY_DATA_HOME", "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/search_r1")
    return os.path.join(data_home, "profile", "multiturn_workload_log", run_id)


def _token_len(tokenizer: PreTrainedTokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _rle(values) -> list[list[int]]:
    runs = []
    last = None
    count = 0
    for value in values:
        value = int(value)
        if last is None:
            last = value
            count = 1
        elif value == last:
            count += 1
        else:
            runs.append([last, count])
            last = value
            count = 1
    if last is not None:
        runs.append([last, count])
    return runs


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    for parser_type, parser_cls in FunctionCallParser.ToolCallParserEnum.items():
        parser = parser_cls()
        if parser.bot_token in tokenizer.get_vocab() and (parser.eot_token == "" or parser.eot_token in tokenizer.get_vocab()):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class AsyncSGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """A SGLang rollout. It requires the module is supported by the SGLang.

        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in SGLang
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        self.do_profile = True
        self._tool_schemas, self._tool_map, self._tool_call_parser_type, self._sgl_tools, self._function_call_parser = self._initialize_tools(config, tokenizer)
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"
        logger.info(f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: {self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: {self._function_call_parser}")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.step=0

    def _collect_workload_enabled(self, is_validate: bool = False) -> bool:
        if self._tp_rank != 0:
            return False
        if is_validate and not _workload_env_flag("WORKLOAD_COLLECT_VALIDATE_MULTITURN", "0"):
            return False
        return _workload_env_flag("WORKLOAD_COLLECT_MULTITURN", "0")

    def _write_workload_jsonl(self, filename: str, record: dict) -> None:
        output_dir = os.getenv("WORKLOAD_LENGTH_DIR") or _workload_default_dir()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _finalize_workload_record(self, record: dict, req: AsyncRolloutRequest, finish_reason_type: FinishReasonTypeEnum, is_validate: bool) -> None:
        if record is None:
            return
        record.update(
            {
                "finish_reason": str(finish_reason_type.value if finish_reason_type is not None else "unknown"),
                "final_total_len": int(len(req.input_ids)),
                "final_response_len": int(len(req.response_ids)),
                "final_response_attention_len": int(sum(req.response_attention_mask)),
                "final_response_loss_len": int(sum(req.response_loss_mask)),
                "response_loss_mask_rle": _rle(req.response_loss_mask),
                "turn_count": int(len(record.get("turns", []))),
                "tool_call_count": int(len(record.get("tool_calls", []))),
                "validate": bool(is_validate),
            }
        )
        self._write_workload_jsonl(f"multiturn_workload_step_{self.step}_rank_{self._rank}.jsonl", record)

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp"))
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length, f"""max_model_len should be greater than total sequence length (prompt_length + response_length): 
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        assert model_hf_config.max_position_embeddings >= self.config.max_model_len, "model context length should be greater than total sequence length"
        # currently max_turns stand for max number of tool calls
        if self.config.multi_turn.max_turns is None:
            self.config.multi_turn.max_turns = self.config.max_model_len // 3

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # log_level="INFO",
                # log_requests=True,
                # log_requests_level=2,
                # max_running_requests=1,
                attention_backend="torch_native",
            )
        else:
            self._engine = None

        self.sharding_manager = None
        # offload
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.get(k)
        self.sampling_params = kwargs

    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool settings
            tokenizer: Tokenizer instance for tool call parsing

        Returns:
            tuple: (tool_schemas, tool_map, tool_call_parser_type, sgl_tools, function_call_parser)
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        import importlib.util
        import sys

        from omegaconf import OmegaConf

        from verl.tools.schemas import OpenAIFunctionToolSchema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.parse_obj(tool_schema_dict)

                tool = tool_cls(config=OmegaConf.to_container(tool_config.config, resolve=True), tool_schema=tool_schema)
                tool_list.append(tool)

            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return tool_schemas, tool_map, tool_call_parser_type, sgl_tools, function_call_parser

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if key in self.sampling_params:
                    old_value = self.sampling_params[key]
                    old_sampling_params_args[key] = old_value
                    self.sampling_params[key] = value
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            self.sampling_params[key] = value

    @GPUMemoryLogger(role="sglang async rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # if self.config.free_cache_engine:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None,
                    }
                )
        else:
            sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # Ensure token IDs are lists
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,  # if validate, already repeat in ray_trainer
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # print(f"{self.sampling_params=}")
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        input_ids=idx_list,
                        image_data=image_list,
                    )
                )
            else:
                output = None
            # Most naive implementation, can extract tensor and send via gloo if too slow
            dist.barrier()
            [output] = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            out = _post_process_outputs(self.tokenizer, output)

            response = out[0].to(idx.device)
            rollout_log_probs = out[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.get("n", 1) > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params["n"], dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params["n"], dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params["n"], dim=0)
                batch_size = batch_size * self.sampling_params["n"]
                _non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    _non_tensor_batch[key] = np.repeat(val, self.sampling_params["n"], axis=0)
            else:
                _non_tensor_batch = non_tensor_batch
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None:
            self._engine.flush_cache()
        self.do_profile=False
        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)

    def _workload_dummy_token_id(self, salt: int = 0) -> int:
        banned = {self.pad_token_id, self.tokenizer.eos_token_id}
        vocab_size = max(2, len(self.tokenizer))
        salt = int(salt)
        if salt:
            candidate = (1000 + salt * 9973) % vocab_size
            for offset in range(32):
                token_id = (candidate + offset) % vocab_size
                if token_id not in banned:
                    return int(token_id)
        token_id = getattr(self, "_workload_dummy_id", None)
        if token_id is not None:
            return int(token_id)
        for text in [" x", " a", " 1", ".", "the"]:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for candidate in ids:
                if candidate not in banned:
                    self._workload_dummy_id = int(candidate)
                    return int(candidate)
        self._workload_dummy_id = 0 if 0 not in banned else 1
        return int(self._workload_dummy_id)

    def _reset_workload_prompt(self, req: AsyncRolloutRequest, target_prompt_len: int, salt: int = 0) -> None:
        target_prompt_len = max(1, min(int(target_prompt_len), int(self.config.prompt_length)))
        token_id = self._workload_dummy_token_id(salt=salt)
        req.prompt_ids = [token_id] * target_prompt_len
        req.input_ids = list(req.prompt_ids)
        req.prompt_attention_mask = [1] * target_prompt_len
        req.attention_mask = list(req.prompt_attention_mask)
        req.prompt_position_ids = list(range(target_prompt_len))
        req.position_ids = list(req.prompt_position_ids)
        req.prompt_loss_mask = [0] * target_prompt_len
        req.loss_mask = list(req.prompt_loss_mask)
        req.response_ids = []
        req.response_attention_mask = []
        req.response_position_ids = []
        req.response_loss_mask = []

    def _append_workload_tokens(self, req: AsyncRolloutRequest, token_count: int, loss_count: int = 0, salt: int = 0) -> None:
        token_count = max(0, int(token_count))
        if token_count == 0:
            return
        loss_count = max(0, min(int(loss_count), token_count))
        token_id = self._workload_dummy_token_id(salt=salt)
        start_position = int(req.position_ids[-1]) + 1 if req.position_ids else 0
        req.input_ids.extend([token_id] * token_count)
        req.attention_mask.extend([1] * token_count)
        req.position_ids.extend(range(start_position, start_position + token_count))
        req.loss_mask.extend([0] * (token_count - loss_count) + [1] * loss_count)

    def _expand_workload_rle(self, rle, target_len: int) -> list[int]:
        target_len = max(0, int(target_len))
        values = []
        if isinstance(rle, list):
            for item in rle:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                value, count = item
                values.extend([int(value)] * max(0, int(count)))
                if len(values) >= target_len:
                    break
        if len(values) < target_len:
            values.extend([1] * (target_len - len(values)))
        return values[:target_len]

    def _workload_sampling_params(self, max_new_tokens: int) -> dict:
        params = dict(self.sampling_params)
        max_new_tokens = max(1, int(max_new_tokens))
        for key, value in {
            "n": 1,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "ignore_eos": True,
        }.items():
            if hasattr(SamplingParams(), key) or key in params:
                params[key] = value
        return params

    def _workload_dummy_queries(self, tool_call: dict) -> list[str]:
        query_count = max(1, int(tool_call.get("query_count", 1)))
        query_text = os.getenv("WORKLOAD_BENCHMARK_QUERY_TEXT", "benchmark query for workload replay")
        return [query_text for _ in range(query_count)]

    async def _replay_workload_tool_call(self, req: AsyncRolloutRequest, tool_call: dict) -> None:
        latency_s = max(0.0, float(tool_call.get("latency_s", 0.0)))
        live_tool = _workload_env_flag("WORKLOAD_BENCHMARK_LIVE_TOOL", "1")
        if live_tool:
            tool_name = str(tool_call.get("tool_name", "search"))
            tool = self._tool_map.get(tool_name)
            start = time.time()
            if tool is not None:
                try:
                    await tool.execute(
                        req.request_id,
                        {"query_list": self._workload_dummy_queries(tool_call)},
                        **req.tools_kwargs.get(tool_name, {}).get("execute_kwargs", {}),
                    )
                except Exception as exc:
                    logger.warning("Benchmark live tool call failed for %s: %s", tool_name, exc)
            elapsed = time.time() - start
            residual_s = latency_s - elapsed
            if _workload_env_flag("WORKLOAD_BENCHMARK_SLEEP_TOOL", "1") and residual_s > 0:
                await asyncio.sleep(residual_s)
        elif _workload_env_flag("WORKLOAD_BENCHMARK_SLEEP_TOOL", "1") and latency_s > 0:
            await asyncio.sleep(latency_s)

    async def _async_replay_workload_request(self, req: AsyncRolloutRequest, do_sample: bool = True, is_validate: bool = False, **kwargs) -> AsyncRolloutRequest:
        workload = req.workload or {}
        target_prompt_len = int(workload.get("prompt_len", len(req.prompt_ids)))
        group_id = int(workload.get("prompt_group_id", workload.get("batch_data_id", 0)))
        sample_id = int(workload.get("sample_id", workload.get("rollout_offset", 0)))
        self._reset_workload_prompt(req, target_prompt_len, salt=group_id + 1)
        if _workload_env_flag("WORKLOAD_BENCHMARK_LIVE_TOOL", "1"):
            await self._handle_pending_state(req)
        req.state = AsyncRolloutRequestStateEnum.RUNNING

        tool_calls_by_turn = {}
        for tool_call in workload.get("tool_calls", []) or []:
            turn = int(tool_call.get("turn", 0))
            tool_calls_by_turn.setdefault(turn, []).append(tool_call)

        for turn in workload.get("turns", []) or []:
            target_generation_prompt_len = int(turn.get("generation_prompt_len", len(req.input_ids)))
            if len(req.input_ids) < target_generation_prompt_len:
                self._append_workload_tokens(req, target_generation_prompt_len - len(req.input_ids), 0, salt=(group_id + 1) * 1000 + sample_id + 1)
            elif len(req.input_ids) > target_generation_prompt_len:
                req.input_ids = req.input_ids[:target_generation_prompt_len]
                req.attention_mask = req.attention_mask[:target_generation_prompt_len]
                req.position_ids = req.position_ids[:target_generation_prompt_len]
                req.loss_mask = req.loss_mask[:target_generation_prompt_len]

            completion_tokens = max(1, int(turn.get("completion_tokens", turn.get("assistant_append_len", 1))))
            sampling_params = self._workload_sampling_params(completion_tokens)
            await self._engine.async_generate(
                input_ids=list(req.input_ids),
                sampling_params=sampling_params,
                return_logprob=False,
            )

            assistant_append_len = int(turn.get("assistant_append_len", completion_tokens))
            assistant_loss_len = int(turn.get("assistant_loss_len", min(completion_tokens, assistant_append_len)))
            turn_id = int(turn.get("turn", 0))
            self._append_workload_tokens(req, assistant_append_len, assistant_loss_len, salt=(group_id + 1) * 100000 + (sample_id + 1) * 100 + turn_id + 1)

            for tool_call in tool_calls_by_turn.get(int(turn.get("turn", 0)), []):
                await self._replay_workload_tool_call(req, tool_call)
                self._append_workload_tokens(req, int(tool_call.get("append_len", 0)), int(tool_call.get("loss_len", 0)), salt=(group_id + 1) * 200000 + (sample_id + 1) * 100 + int(turn.get("turn", 0)) + 1)

        target_response_len = int(workload.get("final_response_len", max(0, len(req.input_ids) - len(req.prompt_ids))))
        target_response_len = max(0, min(target_response_len, int(self.config.response_length)))
        target_total_len = len(req.prompt_ids) + target_response_len
        if len(req.input_ids) < target_total_len:
            self._append_workload_tokens(req, target_total_len - len(req.input_ids), 0, salt=(group_id + 1) * 300000 + sample_id + 1)
        elif len(req.input_ids) > target_total_len:
            req.input_ids = req.input_ids[:target_total_len]
            req.attention_mask = req.attention_mask[:target_total_len]
            req.position_ids = req.position_ids[:target_total_len]
            req.loss_mask = req.loss_mask[:target_total_len]

        response_loss_mask = self._expand_workload_rle(workload.get("response_loss_mask_rle", []), target_response_len)
        req.response_ids = req.input_ids[len(req.prompt_ids) : len(req.prompt_ids) + target_response_len]
        req.response_attention_mask = [1] * target_response_len
        req.response_position_ids = list(range(len(req.prompt_ids), len(req.prompt_ids) + target_response_len))
        req.response_loss_mask = response_loss_mask
        req.prompt_attention_mask = [1] * len(req.prompt_ids)
        req.prompt_position_ids = list(range(len(req.prompt_ids)))
        req.prompt_loss_mask = [0] * len(req.prompt_ids)
        req.attention_mask = req.prompt_attention_mask + req.response_attention_mask
        req.position_ids = req.prompt_position_ids + req.response_position_ids
        req.loss_mask = req.prompt_loss_mask + req.response_loss_mask
        req.reward_scores = {"search": 0.0}
        if _workload_env_flag("WORKLOAD_BENCHMARK_LIVE_TOOL", "1"):
            for name, tool in self._tool_map.items():
                try:
                    await tool.release(req.request_id, **req.tools_kwargs.get(name, {}).get("release_kwargs", {}))
                except Exception as exc:
                    logger.warning("Benchmark live tool release failed for %s: %s", name, exc)
        req.state = AsyncRolloutRequestStateEnum.COMPLETED
        return req

    async def _async_rollout_a_request(self, req: AsyncRolloutRequest, do_sample: bool = True, is_validate: bool = False, **kwargs) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        if _req.workload is not None and _workload_env_flag("WORKLOAD_BENCHMARK", "0"):
            return await self._async_replay_workload_request(_req, do_sample, is_validate, **kwargs)
        finish_reason_type = None
        output = None

        current_turns = 0
        workload_record = None
        if self._collect_workload_enabled(is_validate=is_validate):
            workload_record = {
                "step": int(self.step),
                "batch_data_id": int(_req.batch_data_id),
                "rollout_offset": int(_req.rollout_offset),
                "prompt_len": int(len(_req.prompt_ids)),
                "max_response_len": int(self.config.response_length),
                "max_model_len": int(self.config.max_model_len),
                "turns": [],
                "tool_calls": [],
            }

        def record_generation_turn(output, content, finish_reason, before_input_len, before_loss_len, tool_call_count):
            if workload_record is None:
                return
            meta_info = output.get("meta_info", {}) if isinstance(output, dict) else {}
            completion_tokens = meta_info.get("completion_tokens")
            generated_token_len = int(completion_tokens) if completion_tokens is not None else _token_len(self.tokenizer, content)
            workload_record["turns"].append(
                {
                    "turn": int(current_turns - 1),
                    "generation_prompt_len": int(output.get("_workload_generation_prompt_len", 0)),
                    "prompt_tokens": int(meta_info.get("prompt_tokens", output.get("_workload_generation_prompt_len", 0))),
                    "cached_tokens": int(meta_info.get("cached_tokens", 0)),
                    "max_new_tokens": int(output.get("_workload_max_new_tokens", self.config.response_length)),
                    "completion_tokens": int(generated_token_len),
                    "generated_text_token_len": int(_token_len(self.tokenizer, content)),
                    "assistant_append_len": int(len(_req.input_ids) - before_input_len),
                    "assistant_loss_len": int(sum(_req.loss_mask) - before_loss_len),
                    "total_len_after": int(len(_req.input_ids)),
                    "finish_reason": str(finish_reason.value if finish_reason is not None else "unknown"),
                    "tool_call_count": int(tool_call_count),
                    "engine_latency_s": float(meta_info.get("e2e_latency", 0.0)),
                }
            )

        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = []
                    tool_call_latencies = []
                    import time
                    for tool_call in parsed_tool_calls:
                        start_time = time.time()
                        result = await self._tool_map[tool_call.function.name].execute(
                            _req.request_id,
                            tool_call.function.arguments,
                            **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                        )
                        end_time = time.time()
                        latency = end_time - start_time
                        tool_call_latencies.append({
                            "step": self.step,
                            "request_id": _req.request_id,
                            "tool_name": tool_call.function.name,
                            "latency": latency
                        })
                        tool_call_results.append(result)
                    # Hard-coded debug logging is disabled. Workload collection should
                    # write configurable JSONL records instead.
                    # latency_log_path = "/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/search-r1/logs/tool_call_latency.log"
                    # with open(latency_log_path, "a") as f:
                    #     for entry in tool_call_latencies:
                    #         f.write(f"Step{entry['step']}, Request ID: {entry['request_id']}, Tool: {entry['tool_name']}, Latency: {entry['latency']:.6f} seconds\n")
                    for i, (tool_call, (resp, reward, metrics)) in enumerate(zip(parsed_tool_calls, tool_call_results)):
                        before_input_len = len(_req.input_ids)
                        before_loss_len = sum(_req.loss_mask)
                        _req.add_tool_response_message(self.tokenizer, resp, (i == len(parsed_tool_calls) - 1), format=self.config.multi_turn.format)
                        tool_append_len = len(_req.input_ids) - before_input_len
                        tool_loss_len = sum(_req.loss_mask) - before_loss_len
                        _req.update_metrics(metrics, tool_call.function.name)
                        if workload_record is not None:
                            arguments = tool_call.function.arguments
                            query_list = arguments.get("query_list", []) if isinstance(arguments, dict) else []
                            latency_s = tool_call_latencies[i]["latency"] if i < len(tool_call_latencies) else 0.0
                            workload_record["tool_calls"].append(
                                {
                                    "turn": int(max(current_turns - 1, 0)),
                                    "tool_name": str(tool_call.function.name),
                                    "query_count": int(len(query_list)) if isinstance(query_list, list) else 0,
                                    "latency_s": float(latency_s),
                                    "response_token_len": int(_token_len(self.tokenizer, resp)),
                                    "append_len": int(tool_append_len),
                                    "loss_len": int(tool_loss_len),
                                    "status": str(metrics.get("status", "unknown")) if isinstance(metrics, dict) else "unknown",
                                    "total_results": int(metrics.get("total_results", 0)) if isinstance(metrics, dict) and metrics.get("total_results") is not None else 0,
                                }
                            )
                        if len(_req.input_ids) >= self.config.max_model_len:
                            break
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
                content = output["text"]
                assistant_before_input_len = len(_req.input_ids)
                assistant_before_loss_len = sum(_req.loss_mask)
                turn_tool_call_count = 0
                # output_path="/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/search-r1/logs/length_7b_instruct/output.log"
                # with open(output_path, "a") as f:
                #     f.write(f"Step{self.step}, Request ID: {_req.request_id}, Output Length: {len(content.split())}\n")
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                # with open("/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/search-r1/logs/length_7b_instruct/debug.log", "a") as f:
                #     f.write(f"Step{self.step}, Request ID: {_req.request_id}, Current Turns: {current_turns}, Finish Reason: {finish_reason_type}\n")
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.tokenizer, content, already_over_long=True, format=self.config.multi_turn.format)
                    record_generation_turn(output, content, finish_reason_type, assistant_before_input_len, assistant_before_loss_len, turn_tool_call_count)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.parameters))
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            turn_tool_call_count = len(parsed_tool_calls)
                            _req.add_assistant_message(
                                self.tokenizer,
                                normed_content,
                                tool_calls=parsed_tool_calls,
                                format=self.config.multi_turn.format,
                            )
                            record_generation_turn(output, normed_content, finish_reason_type, assistant_before_input_len, assistant_before_loss_len, turn_tool_call_count)
                        else:
                            _req.add_assistant_message(self.tokenizer, content, format=self.config.multi_turn.format)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            record_generation_turn(output, content, finish_reason_type, assistant_before_input_len, assistant_before_loss_len, turn_tool_call_count)
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(self.tokenizer, content, format=self.config.multi_turn.format)
                        record_generation_turn(output, content, finish_reason_type, assistant_before_input_len, assistant_before_loss_len, turn_tool_call_count)
                        break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)
        self._finalize_workload_record(workload_record, _req, finish_reason_type, is_validate)

        return _req

    async def _handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs) -> dict:
        generation_prompt_ids = _req.get_generation_prompt(self.tokenizer)
        # save_path = "/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/search-r1/logs/length_7b_instruct/input.log"
        # with open(save_path, "a") as f:
        #     f.write(f"Step{self.step}, Request ID: {_req.request_id}, Input Length: {len(generation_prompt_ids)}\n")
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        kwargs["max_new_tokens"] = max_new_tokens
        if "n" not in kwargs or kwargs["n"] > 1:  # group size is supported in preprocess
            kwargs["n"] = 1
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = await self._engine.async_generate(
                input_ids=generation_prompt_ids,
                sampling_params=self.sampling_params,
                return_logprob=False,
            )
        if isinstance(output, dict):
            output["_workload_generation_prompt_len"] = int(len(generation_prompt_ids))
            output["_workload_max_new_tokens"] = int(max_new_tokens)
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        if _req.tools is not None:
            tool_creation_coroutines = []
            for tool_schema in _req.tools:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
            await asyncio.gather(*tool_creation_coroutines)

    @GPUMemoryLogger(role="sglang async rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device
        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
                n=1 if is_validate else self.config.n,
            )
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                )
            )
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None
        dist.barrier()
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), f"""Request {req.request_id} has different length of 
                {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {len(req.input_ids)}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.tokenizer.decode(req.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(req.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(req.response_ids)}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert len(req.input_ids) <= self.config.max_model_len, error_message

            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            if len(req.response_ids) > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {len(req.response_ids)} 
                    greater than max_response_len {self.config.response_length},\n{req=}"""
                )
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(sorted_output_req_list),
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None and self._tp_rank == 0:
            self._engine.flush_cache()
        self.step = self.step + 1
        return DataProto(batch=batch, non_tensor_batch={"messages": np.array(messages), "reward_scores": np.array(reward_scores)})

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        workload_groups = prompts.non_tensor_batch.get("multiturn_workload")
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    _tool_schemas = []
                    for k in _tools_kwargs.keys():
                        _tool_schemas.append(self._tool_map[k].get_openai_tool_schema())
                    prompt_with_chat_template = self.tokenizer.apply_chat_template(
                        conversation=raw_prompt,
                        tools=[tool.model_dump() for tool in _tool_schemas],
                        add_generation_prompt=True,
                        tokenize=False,
                        return_tensors="pt",
                    )
                    input_data = self.tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
                    _input_ids = input_data["input_ids"][0].tolist()
                    _attention_mask = input_data["attention_mask"][0].tolist()
                    _position_ids = compute_position_id_with_mask(input_data["attention_mask"][0]).tolist()
                    if len(_input_ids) > self.config.prompt_length:
                        logger.warning(
                            "Prompt {} has length {} greater than max_prompt_len {}",
                            data_idx,
                            len(_input_ids),
                            self.config.prompt_length,
                        )
                        _input_ids = _input_ids[: self.config.prompt_length]
                        _attention_mask = _attention_mask[: self.config.prompt_length]
                        _position_ids = _position_ids[: self.config.prompt_length]
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                    _position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
                    _tool_schemas = []
                    _tools_kwargs = {}

                workload = None
                if workload_groups is not None:
                    workload_group = workload_groups[data_idx]
                    if isinstance(workload_group, np.ndarray):
                        workload_group = workload_group.tolist()
                    if isinstance(workload_group, (list, tuple)) and len(workload_group) > 0:
                        workload = workload_group[min(rollout_offset, len(workload_group) - 1)]
                    elif isinstance(workload_group, dict):
                        workload = workload_group

                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=[Message.model_validate(msg) for msg in raw_prompt],
                    tools=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    prompt_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    prompt_attention_mask=_attention_mask,
                    response_attention_mask=[],
                    position_ids=_position_ids,
                    prompt_position_ids=_position_ids,
                    response_position_ids=[],
                    loss_mask=[0] * len(_input_ids),
                    prompt_loss_mask=[0] * len(_input_ids),
                    response_loss_mask=[],
                    reward_scores={},
                    workload=workload,
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                )

                error_message = f"Request {req.request_id} has mismatched lengths: input_ids={len(req.input_ids)}, attention_mask={len(req.attention_mask)}, position_ids={len(req.position_ids)}, loss_mask={len(req.loss_mask)}"
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), error_message

                req_list.append(req)

        return req_list

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "chat_completion":
            json_request = args[0]

            formatted_messages = []
            for msg in json_request["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_messages.append(f"{role}: {content}")
            prompt_str = "\n".join(formatted_messages)

            sampling_params_dict = {
                "n": json_request.get("n", 1),
                "max_new_tokens": json_request.get("max_completion_tokens", self.config.response_length),
                "temperature": json_request.get("temperature", 1.0),
                "top_p": json_request.get("top_p", 1.0),
            }
            output = None
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=prompt_str,
                        sampling_params=sampling_params_dict,
                        return_logprob=True,
                    )
                )
            dist.barrier()
            output = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )

            # only return value from master rank
            if self._tp_rank != 0:
                return None
            # build openai chat completion format
            choices = []
            id = None
            for i, content in enumerate(output):
                choices.append(
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": content["text"],
                        },
                        "finish_reason": content["meta_info"]["finish_reason"]["type"],
                    }
                )
                id = content["meta_info"]["id"]

            return {
                "id": "chatcmpl-" + id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": json_request.get("model", "sglang_model"),
                "choices": choices,
            }
        else:
            raise ValueError(f"not supported method : {method}")

        # this function is left for uniform train-inference resharding

    def resume(self):
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801

        self.is_sleep = False

    # this function is left for uniform train-inference resharding
    def offload(self):
        if self.is_sleep:
            return

        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True
