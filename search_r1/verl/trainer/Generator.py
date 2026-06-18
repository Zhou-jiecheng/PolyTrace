import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


def _left_pad_to_length(input_ids, attention_mask, max_length, pad_token_id):
    seq_len = int(input_ids.size(1))
    if seq_len > max_length:
        input_ids = input_ids[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        return input_ids, attention_mask
    if seq_len == max_length:
        return input_ids, attention_mask
    pad_len = max_length - seq_len
    input_pad = torch.full((input_ids.size(0), pad_len), int(pad_token_id), dtype=input_ids.dtype, device=input_ids.device)
    mask_pad = torch.zeros((attention_mask.size(0), pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_pad, input_ids], dim=1), torch.cat([mask_pad, attention_mask], dim=1)


class Generator:
    def __init__(self, file_path, model_path, max_prompt_length=4096):
        self.file_path = file_path
        self.max_prompt_length = int(max_prompt_length)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._piece = None

    def _step_file(self, step):
        step = str(step)
        if os.path.isdir(self.file_path):
            path = os.path.join(self.file_path, f"multiturn_workload_step_{step}.jsonl")
            if os.path.exists(path):
                return path
        if os.path.isfile(self.file_path):
            return self.file_path
        raise FileNotFoundError(f"Cannot find multiturn workload step {step} under {self.file_path}")

    def _load_groups(self, step, n_samples):
        path = self._step_file(step)
        groups = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                if record.get("validate", False):
                    continue
                group_id = int(record.get("prompt_group_id", record.get("batch_data_id", 0)))
                groups[group_id].append(record)
        normalized = []
        for group_id in sorted(groups):
            records = sorted(groups[group_id], key=lambda item: int(item.get("sample_id", item.get("rollout_offset", 0))))
            if not records:
                continue
            if len(records) < n_samples:
                records = records + [records[-1]] * (n_samples - len(records))
            normalized.append(records[:n_samples])
        if not normalized:
            raise ValueError(f"No benchmark workload records found in {path} for step={step}")
        print(f"Loaded multiturn workload step {step} from {path}: prompts={len(normalized)}, samples={sum(len(x) for x in normalized)}")
        return normalized

    def _sample_group_indices(self, groups, target_count):
        indices = list(range(len(groups)))
        if target_count == len(indices):
            return indices
        if target_count > len(indices):
            return indices + random.choices(indices, k=target_count - len(indices))
        return random.sample(indices, target_count)

    def _single_token_piece(self):
        if self._piece is not None:
            return self._piece
        for piece in [" x", " a", " 1", "."]:
            if len(self.tokenizer.encode(piece, add_special_tokens=False)) == 1:
                self._piece = piece
                return piece
        self._piece = " x"
        return self._piece

    def _build_prompt(self, target_prompt_len):
        target_prompt_len = max(1, min(int(target_prompt_len), self.max_prompt_length))
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Answer the question using search when useful."},
        ]
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids, attention_mask = _left_pad_to_length(input_ids, attention_mask, self.max_prompt_length, self.tokenizer.pad_token_id)
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)[-self.max_prompt_length:]
        return messages, raw_prompt_ids, input_ids, attention_mask, position_ids

    def generate(self, bsz=256, step="0", n_samples=8):
        bsz = int(bsz)
        n_samples = int(n_samples)
        groups = self._load_groups(step=step, n_samples=n_samples)
        sampled_indices = self._sample_group_indices(groups, bsz)
        data = {}
        for row_idx, group_idx in enumerate(sampled_indices):
            workload_group = groups[group_idx]
            prompt_len = int(workload_group[0].get("prompt_len", 1))
            raw_prompt, raw_prompt_ids, input_ids, attention_mask, position_ids = self._build_prompt(prompt_len)
            item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "data_source": "search_r1_benchmark",
                "reward_model": {"ground_truth": "benchmark", "style": "rule"},
                "extra_info": {"index": row_idx, "split": "benchmark", "workload_group_id": int(workload_group[0].get("prompt_group_id", group_idx))},
                "index": row_idx,
                "raw_prompt_ids": raw_prompt_ids,
                "raw_prompt": raw_prompt,
                "tools_kwargs": {"search": {"create_kwargs": {}, "execute_kwargs": {}, "calc_reward_kwargs": {}, "release_kwargs": {}}},
                "multiturn_workload": workload_group,
            }
            if not data:
                data = {key: [value] if key not in {"input_ids", "attention_mask", "position_ids"} else value for key, value in item.items()}
            else:
                data["input_ids"] = torch.cat((data["input_ids"], input_ids), dim=0)
                data["attention_mask"] = torch.cat((data["attention_mask"], attention_mask), dim=0)
                data["position_ids"] = torch.cat((data["position_ids"], position_ids), dim=0)
                for key, value in item.items():
                    if key not in {"input_ids", "attention_mask", "position_ids"}:
                        data[key].append(value)
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = np.array(value, dtype=object)
        return DataProto.from_single_dict(data)
