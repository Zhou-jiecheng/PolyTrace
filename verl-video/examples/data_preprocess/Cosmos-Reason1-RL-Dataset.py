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
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets

# from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl-video/data/Processed-Cosmos-Reason1-RL-Dataset")
    parser.add_argument("--data_source", default="/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl-video/data/Cosmos-Reason1-RL-Dataset/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for video processing")
    parser.add_argument("--min_frames", type=int, default=1, help="Minimum number of frames for video processing")
    parser.add_argument("--max_frames", type=int, default=32, help="Maximum number of frames for video processing")

    args = parser.parse_args()
    # configs = ['bridgev2', 'robovqa', 'agibot', 'holoassist']

    # all_datasets = []
    # for config in configs:
    #     dataset = datasets.load_dataset(args.data_source, config)["rl"]
    #     all_datasets.append(dataset)

    # train_dataset = datasets.concatenate_datasets(all_datasets)
    train_dataset = datasets.load_dataset(args.data_source, "robovqa")["rl"]

    # Manually split into train and test (e.g., 90% train, 10% test)
    train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            qa_pairs = example.pop("qa_pairs")
            video = f'/cpfs01/user/zhoujiecheng/workload_rl_analyse/verl-video/data/Cosmos-Reason1-RL-Dataset/{example.pop("video")}'
            problem = qa_pairs["question"] + "\n".join([f"{k}: {v}" for k, v in qa_pairs["index2ans"].items()])
            video_placeholer = "<video>"
            prompt = video_placeholer + "\n" + problem + " " + instruction_following
            answer = qa_pairs["answer"].strip()

            return {
                "data_source": args.data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "videos": [{
                    "type": "video",
                    "video": video,
                    "fps": args.fps,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames
                }],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=64)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=64)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_robovqa_4.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_robovqa_4.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)
