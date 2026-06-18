import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils import hf_tokenizer
from verl.utils.model import compute_position_id_with_mask


class Generator:
    def __init__(self, file_path, model_path, max_prompt_length=2048):
        self.file_path = file_path
        self.max_prompt_length = int(max_prompt_length)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._piece = None

    def _packed_step_from_path(self, path):
        basename = os.path.basename(path)
        prefix = "packed_lengths_step_"
        suffix = ".jsonl"
        if basename.startswith(prefix) and basename.endswith(suffix):
            return basename[len(prefix) : -len(suffix)]
        return None

    def _get_packed_length_info(self, step):
        step = str(step)
        direct_path = os.path.join(self.file_path, f"packed_lengths_step_{step}.jsonl")
        if os.path.exists(direct_path):
            files = [direct_path]
        else:
            files = []
            for root, _, names in os.walk(self.file_path):
                for name in names:
                    if name.startswith("packed_lengths_step_") and name.endswith(".jsonl"):
                        files.append(os.path.join(root, name))
            files.sort(key=lambda path: (self._packed_step_from_path(path) or "", path))

        records = []
        for file_path in files:
            fallback_step = self._packed_step_from_path(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {file_path}:{line_no}: {exc}") from exc
                    if record.get("validate", False):
                        continue
                    if str(record.get("step", fallback_step)) == step:
                        records.append(record)

        if not records:
            print(f"Step {step} not found in packed workload dir {self.file_path}.")
            return None, None

        records.sort(
            key=lambda item: (
                int(item.get("dp_rank", 0)),
                int(item.get("rank", 0)),
                int(item.get("local_index", 0)),
            )
        )
        input_len = [int(item["input"]) for item in records]
        output_len = [[int(value) for value in item.get("output", [])] for item in records]
        print(
            f"Loaded packed workload step {step} from {self.file_path}: "
            f"input={len(input_len)}, output_groups={len(output_len)}, "
            f"outputs={sum(len(group) for group in output_len)}"
        )
        return input_len, output_len

    def get_length_info(self, step):
        if os.path.isdir(self.file_path):
            return self._get_packed_length_info(step)

        step = str(step)
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if step not in data:
            print(f"Step {step} not found in the data.")
            return None, None
        input_len = data[step].get("input", [])
        output_len = data[step].get("output_groups", data[step].get("output", []))
        return input_len, output_len

    def _normalize_output_group(self, output_len, n_samples):
        if isinstance(output_len, np.ndarray):
            output_group = output_len.reshape(-1).tolist()
        elif isinstance(output_len, (list, tuple)):
            output_group = list(output_len)
        else:
            output_group = [output_len]

        output_group = [int(value) for value in output_group]
        if len(output_group) >= n_samples:
            return output_group[:n_samples]
        if not output_group:
            output_group = [1]
        return output_group + [output_group[-1]] * (n_samples - len(output_group))

    def _build_input_output_groups(self, input_lengths, output_lengths, n_samples):
        if not input_lengths or not output_lengths:
            return []

        first_output = output_lengths[0]
        if isinstance(first_output, (list, tuple, np.ndarray)):
            group_count = min(len(input_lengths), len(output_lengths))
            return [
                (int(input_lengths[i]), self._normalize_output_group(output_lengths[i], n_samples))
                for i in range(group_count)
            ]

        complete_output_groups = len(output_lengths) // n_samples
        group_count = min(len(input_lengths), complete_output_groups)
        return [
            (
                int(input_lengths[i]),
                self._normalize_output_group(output_lengths[i * n_samples : (i + 1) * n_samples], n_samples),
            )
            for i in range(group_count)
        ]

    def _sample_group_indices(self, input_lengths, target_count):
        indices = list(range(len(input_lengths)))
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

    def _build_prompt(self, target_input_len):
        target_input_len = max(1, min(int(target_input_len), self.max_prompt_length))
        system = "You are a helpful assistant."
        question_prefix = "Solve the following math problem. Give the final answer in \\boxed{}. Problem:"
        filler = self._single_token_piece()
        best = None
        piece_count = max(0, target_input_len - 32)

        for _ in range(10):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question_prefix + filler * piece_count},
            ]
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            actual_len = int(model_inputs["attention_mask"].sum().item())
            best = (abs(actual_len - target_input_len), actual_len, raw_prompt, model_inputs)
            diff = target_input_len - actual_len
            if diff == 0:
                break
            piece_count = max(0, piece_count + diff)

        _, actual_len, raw_prompt, model_inputs = best
        if actual_len != target_input_len:
            print(f"Warning: synthetic input length target={target_input_len}, actual={actual_len}")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        return raw_prompt, raw_prompt_ids, input_ids, attention_mask, position_ids

    def generate(self, bsz=32, step="0", n_samples=1):
        bsz = int(bsz)
        n_samples = int(n_samples)
        org_input_len, org_output_len = self.get_length_info(step)
        groups = self._build_input_output_groups(org_input_len, org_output_len, n_samples)
        if not groups:
            raise ValueError(f"Step {step} has no complete input/output groups for n_samples={n_samples}.")

        sampled_indices = self._sample_group_indices([input_len for input_len, _ in groups], bsz)
        data = {}
        for row_idx, group_idx in enumerate(sampled_indices):
            input_len, output_len = groups[group_idx]
            raw_prompt, raw_prompt_ids, input_ids, attention_mask, position_ids = self._build_prompt(input_len)
            item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "data_source": "math_dapo",
                "reward_model": {"ground_truth": "0", "style": "rule"},
                "extra_info": {"index": row_idx, "answer": "0", "split": "benchmark"},
                "index": row_idx,
                "tools_kwargs": {},
                "raw_prompt_ids": raw_prompt_ids,
                "raw_prompt": raw_prompt,
                "output_len": output_len,
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
