# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

        profile_dir = os.getenv("VERL_TORCH_PROFILER_DIR","/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl/profile/training_trace")
        if_profile = int(os.getenv("VERL_PROFILE",0))
        # 为profile_dir增加一个时间戳文件夹
        # 确保目录存在
        os.makedirs(profile_dir, exist_ok=True)
        if if_profile == 1:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profile_dir, use_gzip=True))
        else:
            self.profiler=None
        
        self.do_profile=False
        self.count = 0       

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _get_micro_batches(self, data: DataProto) -> Tuple[list, list | None]:
        micro_batch_size_config = data.meta_info.get("micro_batch_size")  # Used for non-dynamic
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)

        batch_tensordict = data.batch  # This is a TensorDict
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

        if has_multi_modal_inputs:
            all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                # Pass the TensorDict part for text-based balancing
                rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                    batch=batch_tensordict, max_token_len=max_token_len
                )

                final_micro_batches_list = []
                for i, text_mb_td in enumerate(rearranged_text_micro_batches_tds):
                    current_original_indices = textual_indices[i]
                    # Gather corresponding multi_modal_inputs
                    current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                    # Create a dictionary for the micro-batch
                    mb_dict = {k: v for k, v in text_mb_td.items()}
                    mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                    final_micro_batches_list.append(mb_dict)
                return final_micro_batches_list, textual_indices
            else:  # Multimodal, but not dynamic_bsz
                # Original logic might have involved data.select().chunk() which returns List[DataProto]
                # data.chunk() splits the DataProto itself.
                num_micro_batches = batch_tensordict.batch_size[0] // micro_batch_size_config
                micro_batches_dp = data.chunk(num_micro_batches)  # Returns List[DataProto]
                return micro_batches_dp, None
        elif use_dynamic_bsz:  # Not multimodal, dynamic_bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches_tds, indices = rearrange_micro_batches(batch=batch_tensordict, max_token_len=max_token_len)
            return micro_batches_tds, indices
        else:  # Not multimodal, not dynamic_bsz
            micro_batches_tds = batch_tensordict.split(micro_batch_size_config)
            return micro_batches_tds, None

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)

        micro_batches_list, dyn_indices = self._get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch_item in micro_batches_list:  # micro_batch_item can be DataProto, dict, or TensorDict
            current_fwd_batch_dict = {}
            if isinstance(micro_batch_item, DataProto):
                current_fwd_batch_dict = {**micro_batch_item.batch, **micro_batch_item.non_tensor_batch}
            elif isinstance(micro_batch_item, dict):  # Handles TensorDict as well as our custom dicts
                current_fwd_batch_dict = micro_batch_item
            else:  # Should be TensorDict if not dict or DataProto
                current_fwd_batch_dict = micro_batch_item  # Assuming it's dict-like (e.g. TensorDict)

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(current_fwd_batch_dict, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy and entropy is not None:
                entropy_lst.append(entropy)

        log_probs_cat = torch.cat(log_probs_lst, dim=0)
        entropys_cat = None
        if calculate_entropy and entropy_lst:
            entropys_cat = torch.cat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            assert dyn_indices is not None, "dyn_indices should not be None when use_dynamic_bsz is True"
            # dyn_indices is List[List[int]], representing the original indices in each micro_batch
            flat_original_indices = list(itertools.chain.from_iterable(dyn_indices))

            # Ensure the number of samples matches before attempting to reorder
            if len(flat_original_indices) == log_probs_cat.size(0):
                revert_indices = torch.tensor(get_reverse_idx(flat_original_indices), dtype=torch.long, device=log_probs_cat.device)
                log_probs_cat = log_probs_cat[revert_indices]
                if entropys_cat is not None:
                    entropys_cat = entropys_cat[revert_indices]
            else:
                logger.warning(
                    f"Dynamic Bsz: Mismatch in reordering inputs for compute_log_prob. "
                    f"Indices count: {len(flat_original_indices)}, LogProbs count: {log_probs_cat.size(0)}. "
                    f"Skipping reordering."
                )

        return log_probs_cat, entropys_cat

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        # if self.do_profile:
        #     torch.cuda.memory._record_memory_history()
        self.actor_module.train()
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        batch_td_for_dataloader = data.batch.select(*select_keys, strict=False)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch_td_for_dataloader.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data_item in enumerate(dataloader):
                # split batch into micro_batches
                micro_batches = []
                if has_multi_modal_inputs:
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = mini_batch_data_item.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = mini_batch_data_item.batch
                        
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )
                    
                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else: # Original non-dynamic multimodal logic
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch_data_item.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = mini_batch_data_item.chunk(num_micro_batches) # Returns List[DataProto]
                elif self.config.use_dynamic_bsz: # Text-only dynamic_bsz
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    # mini_batch_data_item is already the TensorDict for the current PPO mini-batch
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch_data_item, max_token_len=max_token_len)
                else: # Text-only non-dynamic-bsz
                    # the number of gradient_accumulation steps
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch_data_item.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                # if self.do_profile:
                #     if self.profiler is not None:
                #         self.profiler.start()
                for micro_batch_item in micro_batches:
                    current_micro_batch_data_on_device = {}
                    if isinstance(micro_batch_item, DataProto): # Non-dynamic multimodal case
                        for k, v in micro_batch_item.batch.items(): # batch is TensorDict
                            current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                        for k, v in micro_batch_item.non_tensor_batch.items():
                            if k == "multi_modal_inputs" and v is not None:
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else:
                                current_micro_batch_data_on_device[k] = v
                    elif isinstance(micro_batch_item, dict): # Dynamic multimodal (new) or text-only dynamic (if rearrange_micro_batches returns dicts)
                        for k, v in micro_batch_item.items():
                            if isinstance(v, torch.Tensor):
                                current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                            elif k == "multi_modal_inputs" and v is not None: # v is List[Dict[str, Tensor]]
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else: # Other non-tensor data (e.g. list of strings, if any)
                                current_micro_batch_data_on_device[k] = v
                    else: # Assuming TensorDict for text-only cases
                          # (non-dynamic or dynamic if rearrange_micro_batches returns TensorDicts)
                        # .to(device) handles all tensors in TensorDict
                        td_on_device = micro_batch_item.to(torch.cuda.current_device())
                        # _forward_micro_batch expects a dict-like interface
                        current_micro_batch_data_on_device = td_on_device

                    responses = current_micro_batch_data_on_device["responses"]
                    response_length = responses.size(1)
                    attention_mask = current_micro_batch_data_on_device["attention_mask"]
                    if multi_turn:
                        response_mask = current_micro_batch_data_on_device["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = current_micro_batch_data_on_device["old_log_probs"]
                    advantages = current_micro_batch_data_on_device["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=current_micro_batch_data_on_device,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy
                    )
                    
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = current_micro_batch_data_on_device["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                # if self.do_profile and self.profiler is not None:
                #     if batch_idx == 0:
                #         self.profiler.stop()
                #         torch.cuda.memory._dump_snapshot("/cpfs01/user/zhoujiecheng/workload_rl_analyse/search-r1/profile/memory_trace/search-r1-7b.pickle")
                #         self.do_profile=False
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
