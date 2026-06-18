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
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_cosmos_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/Cosmos-Reason1-RL-Dataset/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_video_rl_dataset():
    from verl.utils import hf_tokenizer, hf_processor
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    local_path = "/cpfs01/user/wangzerui/verl/Video-R1/Qwen2.5-VL-7B-COT-SFT"
    hf_processor = hf_processor(local_path)
    tokenizer = hf_tokenizer(local_path)
    local_path = get_cosmos_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config, processor=hf_processor)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    assert "multi_modal_data" in data_proto.non_tensor_batch
    assert "multi_modal_inputs" in data_proto.non_tensor_batch

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")

if __name__ == "__main__":
    test_video_rl_dataset()