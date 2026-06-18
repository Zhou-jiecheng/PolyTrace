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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _env_flag(self, name, default="0"):
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _workload_filter_indices_file(self):
        file_path = os.getenv("WORKLOAD_FILTER_INDICES_FILE", "").strip()
        if file_path:
            return file_path
        length_dir = os.getenv("WORKLOAD_LENGTH_DIR", "").strip()
        if length_dir:
            return os.path.join(length_dir, "dapo_filter_indices.jsonl")
        return None

    def _dump_workload_filter_indices(
        self,
        workload_step,
        trainer_step,
        gen_batch_index,
        prompt_count,
        kept_prompt_indices,
        used_prompt_indices,
        num_prompt_in_batch_before,
    ):
        if not self._env_flag("WORKLOAD_COLLECT_FILTER_INDICES", os.getenv("WORKLOAD_COLLECT_LENGTHS", "0")):
            return
        file_path = self._workload_filter_indices_file()
        if not file_path:
            return
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        record = {
            "workload_step": int(workload_step),
            "trainer_step": int(trainer_step),
            "gen_batch_index": int(gen_batch_index),
            "prompt_count": int(prompt_count),
            "n_sampling": int(self.config.actor_rollout_ref.rollout.n),
            "train_batch_size": int(self.config.data.train_batch_size),
            "num_prompt_in_batch_before": int(num_prompt_in_batch_before),
            "kept_prompt_count": int(len(kept_prompt_indices)),
            "used_prompt_count": int(len(used_prompt_indices)),
            "kept_prompt_indices": [int(idx) for idx in kept_prompt_indices],
            "used_prompt_indices": [int(idx) for idx in used_prompt_indices],
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        print(
            f"Dumped DAPO filter indices: workload_step={workload_step}, "
            f"kept={len(kept_prompt_indices)}, used={len(used_prompt_indices)}, file={file_path}"
        )

    def _load_workload_filter_indices(self):
        file_path = self._workload_filter_indices_file()
        if not file_path or not os.path.exists(file_path):
            return {}
        records = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path}:{line_no}: {exc}") from exc
                records[str(record["workload_step"])] = record
        print(f"Loaded DAPO filter indices from {file_path}: {len(records)} workload steps")
        return records

    def _prompt_indices_to_traj_indices(self, prompt_indices, n_samples):
        traj_indices = []
        for prompt_idx in prompt_indices:
            start = int(prompt_idx) * n_samples
            traj_indices.extend(range(start, start + n_samples))
        return traj_indices

    def benchmark_fit(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not inputs:
            raise ValueError("benchmark_fit requires at least one workload batch")

        metrics = {}
        timing_raw = defaultdict(float)
        generated_batches = []
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        filter_counts = []
        filter_counts_env = os.getenv("WORKLOAD_FILTER_PROMPT_COUNTS", "").strip()
        if filter_counts_env:
            filter_counts = [int(item.strip()) for item in filter_counts_env.split(",") if item.strip()]
            print(f"Benchmark DAPO filter prompt counts fallback: {filter_counts}")
        workload_steps = [
            item.strip()
            for item in os.getenv("WORKLOAD_STEPS", os.getenv("WORKLOAD_STEP", "0")).split(",")
            if item.strip()
        ]
        filter_records = self._load_workload_filter_indices()

        with _timer("step", timing_raw):
            for gen_idx, source_batch in enumerate(inputs):
                new_batch: DataProto = source_batch
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "output_len" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("output_len")

                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                with _timer("gen", timing_raw):
                    print(f"!!!!!!!!!!!!!! benchmark generation start {gen_idx + 1}/{len(inputs)} !!!!!!!!!!!!!!")
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=n_samples, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

                with _timer("reward", timing_raw):
                    print(f"!!!!!!!!!!!!!!!!!! benchmark reward {gen_idx + 1}/{len(inputs)} !!!!!!!!!!!!!!!!!!!!")
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                        new_batch = new_batch.union(reward_tensor)

                    reward_extra_infos_dict = {}
                    try:
                        reward_result = self.reward_fn(new_batch, return_dict=True)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                    except Exception as e:
                        print(f"Error in benchmark reward_fn: {e}")
                        reward_tensor = self.reward_fn(new_batch)

                    new_batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if self.config.algorithm.use_kl_in_reward:
                        new_batch, kl_metrics = apply_kl_penalty(
                            new_batch,
                            kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty,
                        )
                        metrics.update(kl_metrics)
                    else:
                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                selected_prompt_indices = None
                if gen_idx < len(workload_steps):
                    filter_record = filter_records.get(str(workload_steps[gen_idx]))
                    if filter_record is not None:
                        selected_prompt_indices = filter_record.get("used_prompt_indices")
                        if selected_prompt_indices is None:
                            selected_prompt_indices = filter_record.get("kept_prompt_indices")

                if selected_prompt_indices is not None:
                    traj_indices = self._prompt_indices_to_traj_indices(selected_prompt_indices, n_samples)
                    traj_indices = [idx for idx in traj_indices if idx < len(new_batch)]
                    print(
                        f"Benchmark DAPO filter index replay {gen_idx + 1}/{len(inputs)}: "
                        f"workload_step={workload_steps[gen_idx]}, prompts={len(selected_prompt_indices)}, "
                        f"trajs={len(traj_indices)}/{len(new_batch)}"
                    )
                    new_batch = new_batch[traj_indices]
                elif gen_idx < len(filter_counts):
                    keep_prompts = max(0, filter_counts[gen_idx])
                    keep_trajs = min(len(new_batch), keep_prompts * n_samples)
                    print(
                        f"Benchmark DAPO filter count replay {gen_idx + 1}/{len(inputs)}: "
                        f"keep_prompts={keep_prompts}, keep_trajs={keep_trajs}/{len(new_batch)}"
                    )
                    new_batch = new_batch[:keep_trajs]
                elif filter_records:
                    print(f"Warning: no DAPO filter indices for benchmark generation {gen_idx + 1}")

                generated_batches.append(new_batch)

            batch = generated_batches[0] if len(generated_batches) == 1 else DataProto.concat(generated_batches)
            traj_bsz = int(self.config.data.train_batch_size) * n_samples
            if len(batch) < traj_bsz:
                print(f"Warning: benchmark replay has only {len(batch)} trajectories, expected {traj_bsz}")
            if len(batch) > traj_bsz:
                batch = batch[:traj_bsz]

            batch.batch["response_mask"] = compute_response_mask(batch)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("old_log_prob", timing_raw):
                print("!!!!!!!!!!!!!!!!!! benchmark recompute old log prob!!!!!!!!!!!!!!!!!!!!")
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                if "entropys" in old_log_prob.batch:
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics.update({"actor/entropy_loss": entropy_loss.detach().item()})
                    old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with _timer("adv", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=n_samples,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )

            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            with _timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        metrics["train/num_gen_batches"] = len(inputs)
        return dict(timing_raw), metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        workload_filter_step = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_inputs" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    prompt_uids = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    new_batch.non_tensor_batch["uid"] = prompt_uids
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        prompt_count = len(prompt_uids)
                        all_prompt_indices = list(range(prompt_count))
                        self._dump_workload_filter_indices(
                            workload_step=workload_filter_step,
                            trainer_step=self.global_steps,
                            gen_batch_index=num_gen_batches,
                            prompt_count=prompt_count,
                            kept_prompt_indices=all_prompt_indices,
                            used_prompt_indices=all_prompt_indices,
                            num_prompt_in_batch_before=0,
                        )
                        workload_filter_step += 1
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        prompt_bsz = self.config.data.train_batch_size
                        num_prompt_in_batch_before = num_prompt_in_batch
                        remaining_prompt_slots = max(0, prompt_bsz - num_prompt_in_batch_before)
                        kept_prompt_uid_set = set(kept_prompt_uids)
                        kept_prompt_indices = [idx for idx, uid in enumerate(prompt_uids) if uid in kept_prompt_uid_set]
                        used_prompt_indices = kept_prompt_indices[:remaining_prompt_slots]
                        self._dump_workload_filter_indices(
                            workload_step=workload_filter_step,
                            trainer_step=self.global_steps,
                            gen_batch_index=num_gen_batches,
                            prompt_count=len(prompt_uids),
                            kept_prompt_indices=kept_prompt_indices,
                            used_prompt_indices=used_prompt_indices,
                            num_prompt_in_batch_before=num_prompt_in_batch_before,
                        )
                        workload_filter_step += 1
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with _timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
