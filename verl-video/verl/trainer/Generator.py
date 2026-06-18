from verl import DataProto
import torch
import numpy as np
import json
import os
import random
import torch.nn.functional as F
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_processor
from verl.utils import hf_tokenizer
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from scipy import stats


class Generator:
    def __init__(self, file_path,model_path):
        self.file_path = file_path
        self.processor = hf_processor(model_path)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    
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
            files = [
                os.path.join(self.file_path, name)
                for name in os.listdir(self.file_path)
                if name.startswith("packed_lengths_step_") and name.endswith(".jsonl")
            ]
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
                    record_step = str(record.get("step", fallback_step))
                    if record_step == step:
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
        """get specific step's input/output length from a workload JSON or packed JSONL dir."""
        if os.path.isdir(self.file_path):
            return self._get_packed_length_info(step)

        step = str(step)
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        print(data.keys())
        if step in data:
            input_len = data[step].get('input', [])
            output_len = data[step].get('output_groups', data[step].get('output', []))
            return input_len, output_len
        else:
            print(f"Step {step} not found in the data.")
            return None, None
        
    def sample_from_range_distribution(self, original_list, target_count):
        """
        sample target count numbers from a range distribution based on the original list.
        """
        if not original_list or target_count <= 0:
            return []
        
        if target_count >= len(original_list):
            if target_count == len(original_list):
                result = original_list.copy()
            else:
                result = random.choices(original_list, k=target_count)
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


    def generate_industrial_workload(self, bsz, type="math"):
        """
        generate industrial workload
        """
        math_input_dist_param = (np.float64(0.5320355020031421), 48.39466670710811, np.float64(80.14650081294441))
        math_output_dist_param = (np.float64(2.4379588100102367), np.float64(195.42315070509), np.float64(3847.2423976371874))
        math_inputlen_list = stats.lognorm.rvs(*math_input_dist_param, size=bsz)
        math_outputlen_list = stats.gamma.rvs(*math_output_dist_param, size=bsz)


    def _build_input_output_groups(self, input_lengths, output_lengths, n_samples):
        if not input_lengths or not output_lengths:
            return []

        first_output = output_lengths[0]
        if isinstance(first_output, (list, tuple)):
            group_count = min(len(input_lengths), len(output_lengths))
            return [
                (input_lengths[i], list(output_lengths[i])[:n_samples])
                for i in range(group_count)
                if len(output_lengths[i]) >= n_samples
            ]

        complete_output_groups = len(output_lengths) // n_samples
        group_count = min(len(input_lengths), complete_output_groups)
        if group_count < len(input_lengths) or group_count * n_samples < len(output_lengths):
            print(
                f"Use {group_count} aligned workload groups, "
                f"ignore {len(input_lengths) - group_count} inputs and "
                f"{len(output_lengths) - group_count * n_samples} outputs."
            )

        return [
            (
                input_lengths[i],
                output_lengths[i * n_samples : (i + 1) * n_samples],
            )
            for i in range(group_count)
        ]

    def sample_group_indices_from_range_distribution(self, input_lengths, target_count):
        if not input_lengths or target_count <= 0:
            return []

        indices = list(range(len(input_lengths)))
        if target_count >= len(indices):
            if target_count == len(indices):
                result = indices.copy()
            else:
                result = random.choices(indices, k=target_count)
            random.shuffle(result)
            return result

        min_val = min(input_lengths)
        max_val = max(input_lengths)
        num_bins = min(20, len(set(input_lengths)))
        bin_size = (max_val - min_val) / num_bins if num_bins > 1 else 1

        if bin_size == 0:
            return random.choices(indices, k=target_count)

        bin_indices = [[] for _ in range(num_bins)]
        for idx, val in enumerate(input_lengths):
            bin_idx = min(int((val - min_val) / bin_size), num_bins - 1)
            bin_indices[bin_idx].append(idx)

        result = []
        for values_in_bin in bin_indices:
            if not values_in_bin:
                continue
            proportion = len(values_in_bin) / len(input_lengths)
            samples_needed = int(proportion * target_count)
            if samples_needed > 0:
                result.extend(random.choices(values_in_bin, k=samples_needed))

        while len(result) < target_count:
            result.append(random.choice(indices))

        if len(result) > target_count:
            result = random.sample(result, target_count)

        random.shuffle(result)
        return result

    def _sample_prompt_and_output_lengths(self, step, num_prompts, n_samples):
        org_step_input_len, org_step_output_len = self.get_length_info(step)
        groups = self._build_input_output_groups(org_step_input_len, org_step_output_len, n_samples)
        if not groups:
            raise ValueError(
                f"Step {step} has no complete input/output groups for n_samples={n_samples}."
            )

        group_input_lengths = [input_len for input_len, _ in groups]
        sampled_indices = self.sample_group_indices_from_range_distribution(group_input_lengths, num_prompts)
        step_input_len = [groups[i][0] for i in sampled_indices]
        step_output_len = [self._normalize_output_group(groups[i][1], n_samples) for i in sampled_indices]

        return step_input_len, step_output_len

    def _normalize_output_group(self, output_len, n_samples):
        if isinstance(output_len, np.ndarray):
            output_group = output_len.reshape(-1).tolist()
        elif isinstance(output_len, (list, tuple)):
            output_group = list(output_len)
        else:
            output_group = [output_len]

        output_group = [int(value) for value in output_group]
        if len(output_group) != n_samples:
            raise ValueError(
                f"Each prompt must own exactly n_samples={n_samples} output lengths, "
                f"but got {len(output_group)}: {output_group}"
            )
        return output_group

    def _single_token_piece(self):
        if hasattr(self, "_cached_single_token_piece"):
            return self._cached_single_token_piece

        candidates = [" a", " the", " in", " to", " of", " and", " is", " for", " on", " robot", " task"]
        for piece in candidates:
            if (
                len(self.tokenizer.encode(piece, add_special_tokens=False)) == 1
                and len(self.tokenizer.encode(piece * 32, add_special_tokens=False)) == 32
            ):
                self._cached_single_token_piece = piece
                return piece

        self._cached_single_token_piece = " a"
        return self._cached_single_token_piece

    def _build_video_raw_prompt(self, text_content):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "\n" + text_content},
                ],
            }
        ]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _build_video_model_inputs(self, text_content, videos):
        raw_prompt = self._build_video_raw_prompt(text_content)
        model_inputs = self.processor(text=[raw_prompt], images=None, videos=videos, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=16000,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")
        return raw_prompt, model_inputs, input_ids, attention_mask

    def _build_target_length_video_sample(self, target_input_len, video_shape):
        vs = video_shape[1:]
        videos = torch.from_numpy(np.random.rand(*vs).astype(np.float32))
        filler = self._single_token_piece()
        piece_count = 0
        best = None

        for _ in range(8):
            text_content = filler * max(piece_count, 0)
            raw_prompt, model_inputs, input_ids, attention_mask = self._build_video_model_inputs(text_content, videos)
            actual_len = int(attention_mask.sum().item())
            best = (abs(actual_len - target_input_len), actual_len, raw_prompt, model_inputs, input_ids, attention_mask)
            diff = int(target_input_len) - actual_len
            if diff == 0:
                return raw_prompt, videos, model_inputs, input_ids, attention_mask
            piece_count = max(0, piece_count + diff)

        _, actual_len, raw_prompt, model_inputs, input_ids, attention_mask = best
        if actual_len != int(target_input_len):
            print(f"Warning: synthetic input length target={target_input_len}, actual={actual_len}")
        return raw_prompt, videos, model_inputs, input_ids, attention_mask

    def generate(self, bsz=64, step="0", n_samples=1):
        """
        A dummy workload generate example. Generate Cosmos workloads
        """
        bsz = int(bsz)
        n_samples = int(n_samples)
        if bsz <= 0:
            raise ValueError(f"bsz must be positive, got {bsz}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        data={}
        step_input_len, step_output_len = self._sample_prompt_and_output_lengths(step, bsz, n_samples)


        for input_len, output_len in zip(step_input_len, step_output_len):
            # Generate one random prompt/video per workload group. The rollout layer
            # expands this group to n_samples by copying the same prompt input.
            output_len = self._normalize_output_group(output_len, n_samples)
            if input_len < 960:
                video_shape = [1, 2, 3, 532, 952]
            elif input_len < 1500:
                video_shape = [1, 4, 3, 532, 952]
            elif input_len < 3500:
                video_shape = [1, 14, 3, 504, 644]
            elif input_len < 4000:
                video_shape = [1, 18, 3, 504, 644]
            else:
                video_shape = [1, 20, 3, 504, 644]

            raw_prompt, videos, model_inputs, input_ids, attention_mask = self._build_target_length_video_sample(
                int(input_len),
                video_shape,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            position_id = position_ids[0].unsqueeze(0)
            input_id = input_ids

            data_source = "/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl-video/data/Cosmos-Reason1-RL-Dataset/"
            ability = "math"
            reward_model = {'ground_truth': 'A', 'style': 'rule'}
            
            extra_info = {'answer': 'A', 
                          'index': random.randint(400,500),
                          'question': 'The agent in the video was given the instruction - pick up the glasses. Is it possible for the agent to execute the task specified in the instruction?A: yes\nB: no', 
                          'split': 'train'}
            index = extra_info['index']
            raw_prompt_id = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            multi_modal_data = {
                'video': [videos.numpy()]
            }
            multi_modal_input = dict(model_inputs)
            if not data:
                data = {
                    'input_ids': input_id,
                    'attention_mask': attention_mask,
                    'position_ids': position_id,
                    'data_source': [data_source],
                    'ability': [ability],
                    'reward_model': [reward_model],
                    'extra_info': [extra_info],
                    'index': [index],
                    'tools_kwargs': [{}],
                    'raw_prompt_ids': [raw_prompt_id],
                    'multi_modal_data': [multi_modal_data],
                    'multi_modal_inputs': [multi_modal_input],
                    'output_len': [output_len],
                }
            else:
                data['input_ids'] = torch.cat((data['input_ids'], input_id), dim=0)
                data['attention_mask'] = torch.cat((data['attention_mask'], attention_mask), dim=0)
                data['position_ids'] = torch.cat((data['position_ids'], position_id), dim=0)
                data['data_source'].append(data_source)
                data['ability'].append(ability)
                data['reward_model'].append(reward_model)
                data['extra_info'].append(extra_info)
                data['index'].append(index)
                data['tools_kwargs'].append({})
                data['raw_prompt_ids'].append(raw_prompt_id)
                data['multi_modal_data'].append(multi_modal_data)
                data['multi_modal_inputs'].append(multi_modal_input)
                data['output_len'].append(output_len)

        for k,v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v, dtype=object)
            else:
                data[k] = v
        # print(data['input_ids'].shape, data['attention_mask'].shape, data['position_ids'].shape)
        return DataProto.from_single_dict(data)

    def generate_text(self, bsz=64, step="0", n_samples=1):
        """
        generate pure text input
        """
        bsz = int(bsz)
        n_samples = int(n_samples)
        if bsz <= 0:
            raise ValueError(f"bsz must be positive, got {bsz}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        data={}
        step_input_len, step_output_len = self._sample_prompt_and_output_lengths(step, bsz, n_samples)
        
        for input_len, output_len in zip(step_input_len, step_output_len):
            # Generate one random prompt per workload group. The rollout layer expands
            # this group to n_samples by copying the same prompt input.
            output_len = self._normalize_output_group(output_len, n_samples)
            vocabulary = [
                "the", "agent", "in", "video", "was", "given", "instruction", "pick", "up",
                "glasses", "is", "it", "possible", "for", "to", "execute", "task",
                "specified", "yes", "no", "a", "b", "c", "d", "question", "answer",
                "robot", "table", "chair", "book", "pen", "apple", "banana", "orange",
                "cup", "bottle", "laptop", "phone", "keys", "remote", "door", "window",
                "box", "ball", "move", "place", "open", "close", "find", "bring", "put",
                "take", "go", "stop", "on", "under", "beside", "next", "from", "into",
                "out", "of", "with", "near", "far", "red", "blue", "green", "yellow",
                "black", "white", "big", "small", "heavy", "light", "fast", "slow",
                "what", "where", "when", "why", "how", "which", "who", "person",
                "object", "item", "location", "command", "request", "action", "perform",
                "complete", "succeed", "fail", "correctly", "incorrectly", "left", "right",
                "forward", "backward"
            ]
            context = random.choices(vocabulary, k=input_len)

            # 构建包含视频标记的 messages 格式（参考 rl_dataset.py）
            text_content = " ".join(context)
            messages = [
                {
                    "role": "user", 
                    "content": f"{text_content}"
                }
            ]
            for message in messages:
                content = message["content"]
                content_list = []
                import re
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
            
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(text=[raw_prompt], images=None, videos=None, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=16000,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="error",
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            
            input_id = input_ids[0].unsqueeze(0)
            attention_mask = attention_mask[0].unsqueeze(0)
            position_id = position_ids[0].unsqueeze(0)
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            data_source = "math500"
            ability = "math"
            reward_model = {'ground_truth': 'A', 'style': 'rule'}
            extra_info = {'answer': 'A', 
                          'index': random.randint(400,500),
                          'question': 'The agent in the video was given the instruction - pick up the glasses. Is it possible for the agent to execute the task specified in the instruction?A: yes\nB: no', 
                          'split': 'train'}
            if not data:
                data = {
                    'input_ids': input_id,
                    'attention_mask': attention_mask,
                    'position_ids': position_id,
                    'data_source': [data_source],
                    'ability': [ability],
                    'reward_model': [reward_model],
                    'extra_info': [extra_info],
                    'index': [extra_info['index']],
                    'tools_kwargs': [{}],
                    'raw_prompt_ids': [raw_prompt_ids],
                    'output_len': [output_len],
                }
            else:
                data['input_ids'] = torch.cat((data['input_ids'], input_id), dim=0)
                data['attention_mask'] = torch.cat((data['attention_mask'], attention_mask), dim=0)
                data['position_ids'] = torch.cat((data['position_ids'], position_id), dim=0)
                data['data_source'].append(data_source)
                data['ability'].append(ability)
                data['reward_model'].append(reward_model)
                data['extra_info'].append(extra_info)
                data['index'].append(extra_info['index'])
                data['tools_kwargs'].append({})
                data['raw_prompt_ids'].append(raw_prompt_ids)
                data['output_len'].append(output_len)
            
        for k,v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v, dtype=object)
            else:
                data[k] = v
        # print(data['input_ids'].shape, data['attention_mask'].shape, data['position_ids'].shape)
        return DataProto.from_single_dict(data)
            
    def process_batch(self, batch):
        """
        处理批量数据的方法。
        """
        processed_batch = []
        for item in batch:
            processed_item = self.generate(item)
            processed_batch.append(processed_item)
        
        return processed_batch

def print_dataproto(batch):
    """
    打印DataProto对象的内容。
    """
    # 如果是DataProto对象，分析其结构
    if hasattr(batch, 'batch'):
        print(f"\n=== batch 字段分析 ===")
        print(f"batch类型: {type(batch.batch)}")
        if hasattr(batch.batch, 'keys'):
            batch_keys = list(batch.batch.keys())
            print(f"batch键: {batch_keys}")
            
            # 分析每个tensor的形状
            for key in batch_keys:
                value = batch.batch[key]
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                
                else:
                    print(f"  {key}: {type(value)}")
    
    if hasattr(batch, 'non_tensor_batch'):
        print(f"\n=== non_tensor_batch 字段分析 ===")
        print(f"non_tensor_batch类型: {type(batch.non_tensor_batch)}")
        if hasattr(batch.non_tensor_batch, 'keys'):
            non_tensor_keys = list(batch.non_tensor_batch.keys())
            print(f"non_tensor_batch键: {non_tensor_keys}")
            
            for key in non_tensor_keys:
                value = batch.non_tensor_batch[key]
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                    
                    # 如果是object数组，分析其内容
                    if value.dtype == 'object':
                        print(f"    详细内容分析:")
                        if len(value) > 0:
                            first_element = value[0]
                            print(f"    - 第一个元素类型: {type(first_element)}")
                            print(f"    - 第一个元素内容: {first_element}")
                            
                            # 分析不同元素是否类型一致
                            unique_types = set(type(item) for item in value)
                            print(f"    - 所有元素类型: {unique_types}")
                            
                            # 对所有字段进行详细分析
                            print(f"    - 详细内容分析 ({key}):")
                            
                            if isinstance(first_element, dict):
                                print(f"      字典类型，键: {list(first_element.keys())}")
                                for k, v in first_element.items():
                                    if hasattr(v, 'shape'):
                                        print(f"        {k}: {type(v)} shape={v.shape}")
                                    elif hasattr(v, '__len__') and not isinstance(v, str):
                                        print(f"        {k}: {type(v)} length={len(v)}")
                                        
                                        # 特别分析video字段的4-D array结构
                                        if k == 'video':
                                            print(f"        🎥 video字段详细分析:")
                                            print(f"          数据类型: {type(v)}")
                                            
                                            # 处理numpy array或类似的多维数组
                                            if hasattr(v, 'shape'):
                                                print(f"          形状: {v.shape}")
                                                print(f"          数据类型: {v.dtype}")
                                                print(f"          维度数: {v.ndim}")
                                                if v.ndim == 4:
                                                    print(f"          维度1 (batch): {v.shape[0]}")
                                                    print(f"          维度2 (frames/time): {v.shape[1]}")
                                                    print(f"          维度3 (height): {v.shape[2]}")
                                                    print(f"          维度4 (width): {v.shape[3]}")
                                                    if v.size > 0:
                                                        print(f"          数值范围: [{v.min():.4f}, {v.max():.4f}]")
                                                        print(f"          平均值: {v.mean():.4f}")
                                                        print(f"          标准差: {v.std():.4f}")
                                                else:
                                                    print(f"          ⚠️  不是4维数组，实际维度: {v.ndim}")
                                                    for i, size in enumerate(v.shape):
                                                        print(f"          维度{i+1}: {size}")
                                            
                                            # 处理其他类型的序列（如object array包含嵌套结构）
                                            elif hasattr(v, '__getitem__') and len(v) > 0:
                                                print(f"          第一个元素类型: {type(v[0])}")
                                                if hasattr(v[0], '__len__'):
                                                    dims = []
                                                    current = v
                                                    for level in range(4):  # 检查最多4个维度
                                                        dims.append(len(current))
                                                        if len(current) > 0:
                                                            first_item = current[0]
                                                            if hasattr(first_item, '__len__') and not isinstance(first_item, str):
                                                                current = first_item
                                                            else:
                                                                print(f"          元素类型: {type(first_item)}")
                                                                break
                                                        else:
                                                            break
                                                    
                                                    print(f"          嵌套维度结构: {dims}")
                                                    if len(dims) >= 4:
                                                        print(f"          维度1 (batch): {dims[0]}")
                                                        print(f"          维度2 (frames/time): {dims[1]}")
                                                        print(f"          维度3 (height): {dims[2]}")
                                                        print(f"          维度4 (width): {dims[3]}")
                                            
                                            # 显示几个样本的维度信息
                                            print(f"          📊 前3个样本的维度信息:")
                                            for i in range(min(3, len(v))):
                                                sample = v[i]
                                                if hasattr(sample, 'shape'):
                                                    print(f"            样本[{i}]: {sample.shape}")
                                                elif hasattr(sample, '__len__'):
                                                    sample_dims = []
                                                    current = sample
                                                    for level in range(3):
                                                        if hasattr(current, '__len__'):
                                                            sample_dims.append(len(current))
                                                            if len(current) > 0:
                                                                current = current[0]
                                                            else:
                                                                break
                                                        else:
                                                            break
                                                    print(f"            样本[{i}]: {sample_dims}")
                                                else:
                                                    print(f"            样本[{i}]: {type(sample)}")
                                    else:
                                        print(f"        {k}: {type(v)} = {v}")
                            
                            elif isinstance(first_element, (list, tuple)):
                                print(f"      序列类型，长度: {len(first_element)}")
                                if len(first_element) > 0:
                                    element_types = set(type(x) for x in first_element[:5])
                                    print(f"        前5个元素类型: {element_types}")
                                    print(f"        前10个元素: {first_element[:10]}")
                            
                            elif isinstance(first_element, str):
                                print(f"      字符串类型，长度: {len(first_element)}")
                                print(f"        内容预览: {first_element[:100]}...")
                            
                            elif isinstance(first_element, (int, float)):
                                try:
                                    import numpy as np
                                    arr = np.array(value)
                                    print(f"      数值类型: {type(first_element)}")
                                    print(f"        范围: {arr.min()} - {arr.max()}")
                                    print(f"        平均值: {arr.mean():.2f}")
                                except:
                                    print(f"      数值类型: {type(first_element)}")
                                    print(f"        前5个值: {value[:5]}")
                            
                            else:
                                print(f"      其他类型: {type(first_element)}")
                                print(f"        值: {first_element}")
                            
                            # 检查数组中是否有不同类型的元素
                            if len(unique_types) > 1:
                                print(f"      ⚠️  数组包含多种类型!")
                                for i, item in enumerate(value[:5]):
                                    print(f"        [{i}]: {type(item)} = {item}")
                            
                            # 显示几个样本
                            print(f"    - 样本数据 (前3个):")
                            for i, item in enumerate(value[:3]):
                                if isinstance(item, dict):
                                    print(f"      [{i}]: dict with keys {list(item.keys())}")
                                    # 显示字典内容概览
                                    for k, v in list(item.items())[:3]:
                                        if hasattr(v, 'shape'):
                                            print(f"          {k}: {type(v)} {v.shape} {v.dtype}")
                                        else:
                                            preview = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                                            print(f"          {k}: {type(v)} = {preview}")
                                elif isinstance(item, (list, tuple)):
                                    print(f"      [{i}]: {type(item).__name__} length={len(item)}")
                                    if len(item) > 0:
                                        print(f"          前3个元素: {item[:3]}")
                                else:
                                    preview = str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                                    print(f"      [{i}]: {type(item)} = {preview}")
                        else:
                            print(f"    - 空数组")
                else:
                    print(f"  {key}: {type(value)}")
                    # 对于非numpy数组的情况
                    if hasattr(value, '__len__'):
                        print(f"    - 长度: {len(value)}")
                        if len(value) > 0:
                            print(f"    - 第一个元素: {type(value[0])} = {value[0]}")
                    
                    
                    
    
    if hasattr(batch, 'meta_info'):
        print(f"\n=== meta_info 字段分析 ===")
        print(f"meta_info: {batch.meta_info}")
    
    # 如果有attention_mask，分析序列长度
    if hasattr(batch, 'batch') and 'attention_mask' in batch.batch:
        attention_mask = batch.batch['attention_mask']
        if hasattr(attention_mask, 'sum'):
            seq_lengths = attention_mask.sum(dim=-1)
            print(f"\n=== 序列长度分析 ===")
            print(f"序列长度: {seq_lengths.tolist()}")
            print(f"最小长度: {seq_lengths.min().item()}")
            print(f"最大长度: {seq_lengths.max().item()}")
            print(f"平均长度: {seq_lengths.float().mean().item():.1f}")


if __name__ == "__main__":
    generator = Generator('/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/verl-video/logs/cosmos_workloads.json',"/cpfs01/shared/llm_s/zhoujiecheng/workload_rl_analyse/models/Qwen2.5-VL-7B-COT-SFT")
    
    data = generator.generate(bsz=2)
    print_dataproto(data)
    special_token = {}
    ids = list(data.batch["input_ids"][0])
    count = 0
    positions = []
    idx = 0
    for pos, val in enumerate(ids):
        if val > 151642:
            positions.append((idx, pos, val))
            if str(val) not in special_token:
                special_token[str(val)] = {"count": 0, "positions": []}
            special_token[str(val)]["count"] += 1
            special_token[str(val)]["positions"].append(pos)
    for key, value in special_token.items():
        print(f"Token ID: {key}, Count: {value['count']}")
    
