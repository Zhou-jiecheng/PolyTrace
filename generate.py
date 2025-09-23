# A example of generating DataProto from PolyTrace
from verl import DataProto
import torch
import numpy as np
import json
import random
import torch.nn.functional as F
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_processor
from verl.utils import hf_tokenizer
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from scipy import stats


class Generator:
    def __init__(self, file_path, model_path, parameters):
        self.file_path = file_path
        self.processor = hf_processor(model_path)
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        self.parameters = parameters
        self.vocabulary = [
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

    def get_length_info(self, step):
        """get specific step's input/output length from a JSON file."""
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        print(data.keys())
        if step in data:
            input_len = data[step].get('input', [])
            output_len = data[step].get('output', [])
            return input_len, output_len
        else:
            print(f"Step {step} not found in the data.")
            return None, None
        
    def sample_from_range_distribution(self, original_list, target_count):
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

    def generate(self, bsz=64, step="0"):
        """
        A dummy workload generate example. Generate Cosmos workloads
        """
        data={}
        org_step_input_len,org_step_output_len = self.get_length_info(step)
        step_input_len = self.sample_from_range_distribution(org_step_input_len, bsz)
        step_output_len = self.sample_from_range_distribution(org_step_output_len, bsz)

        # TODO: support different vision encoders and video data, this just a hardcore for cosmos dataset and qwen2.5-vl
        for input_len, output_len in zip(step_input_len, step_output_len):
            if input_len < 960:
                video_len = 480
                video_grid_thw = [1, 32, 60]
                pixel_values_shape = [1920, 1176]
                video_shape = [1, 4, 3, 532, 952]
                seq_len = input_len - video_len
            
            elif input_len < 1500:
                video_len = 960
                video_grid_thw = [2, 32, 60]
                pixel_values_shape = [3840, 1176]
                video_shape = [1, 4, 3, 532, 952]
                seq_len = input_len - video_len
            elif input_len < 3500:
                video_len = 2898
                video_grid_thw = [7, 36, 46]
                pixel_values_shape = [11592, 1176]
                video_shape = [1, 14, 3, 504, 644]
                seq_len = input_len - video_len
            elif input_len < 4000:
                video_len = 3726
                video_grid_thw =[9, 36, 46]
                pixel_values_shape = [14904, 1176]
                video_shape = [1, 18, 3, 504, 644]
                seq_len = input_len - video_len
            elif input_len < 5000:
                video_len = 4140
                video_grid_thw = [10, 36, 46]
                pixel_values_shape = [16560, 1176]                
                video_shape =  [1, 20, 3, 504, 644]
                seq_len = input_len - video_len
            else:
                video_len = 4140
                video_grid_thw = [10, 36, 46]
                pixel_values_shape = [16560, 1176]                
                video_shape =  [1, 20, 3, 504, 644]
                seq_len = input_len - video_len        

            context = random.choices(self.vocabulary, k=seq_len)

            # Align with rl_dataset.py data format
            text_content = " ".join(context)
            messages = [
                {
                    "role": "user", 
                    "content": f"<video>\n{text_content}"
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
            
            vs = video_shape[1:]
            
            # Generate random pixel values in [0, 1] range and cast to float32
            videos = np.random.rand(*vs).astype(np.float32)
            videos = torch.from_numpy(videos)
            print(f"videos shape: {videos.shape}, dtype: {videos.dtype}")
            print(f"videos value range: [{videos.min():.4f}, {videos.max():.4f}]")
            print(f"raw_prompt: {raw_prompt[:200]}...") 
            
            model_inputs = self.processor(text=[raw_prompt], images=None, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            print(model_inputs.keys())

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
                

            position_ids = compute_position_id_with_mask(attention_mask)
            position_id = position_ids[0]
            position_id = position_id.unsqueeze(0)
            
            input_id = input_ids
            print(f"input_id shape: {input_id.shape}, attention_mask shape: {attention_mask.shape}, position_id shape: {position_id.shape}")

            data_source = "video_understanding" # align with reward compute method
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
        print(data['input_ids'].shape, data['attention_mask'].shape, data['position_ids'].shape)
        return DataProto.from_single_dict(data)

    def generate_text(self, bsz=64, step="0"):
        """
        generate pure text input
        """
        data={}
        org_step_input_len,org_step_output_len = self.get_length_info(step)
        step_input_len = self.sample_from_range_distribution(org_step_input_len, bsz)
        step_output_len = self.sample_from_range_distribution(org_step_output_len, bsz)
        
        for input_len, output_len in zip(step_input_len, step_output_len):
            context = random.choices(self.vocabulary, k=input_len)

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
        print(data['input_ids'].shape, data['attention_mask'].shape, data['position_ids'].shape)
        return DataProto.from_single_dict(data)


if __name__ == "__main__":
    generator = Generator('./PolyTrace/cosmos_workloads.json',"Qwen/Qwen2.5-VL")
    
    data = generator.generate(bsz=256)
    special_token = {}
    ids = list(data.batch["input_ids"][0])
    count = 0
    positions = []
    idx = 0