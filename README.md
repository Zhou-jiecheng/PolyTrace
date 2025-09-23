## PolyTrace

This repository hosts the public releases of PolyTrace in LLM RLVR training. Containing 3 large scale and in house RL training workloads and 4 open source RL training workload. Meanly contains 2 kind of data, tool call latency and realistic workloads.

Both of them continuously enhancing for more tasks.


| Task                    | Model size | Total Step | Dataset        |
|-------------------------|------------|------------|----------------|
| Mathematics             | 32B   | 191        | DAPO     |
| Programming             | 14B    | 190        | DeepCoder |
| Searching               | 7B    | 147        | NO hotpotqa |
| Video Understanding     | 7B     | 112        | RoboVQA    |
| Mathematics             | 235B       | 188       | In house       |
| Tool Use                | 235B       | 59        | In house       |
| Image Understanding     | 235B       | 46        | In house       |

**Table 4: Summary of PolyTrace.**

## Data structure

We collect the length from training trajectory for data anonymization.

```
{
    "0":
        {
            "input":[],
            "output":[],
        },
    "1":
        {
            "input":[],
            "output":[],
        }
    ...
}
```

For multi-turn task workloads:

```
{
    "0":[
            {
                "input":[...],
                "output":[...],
            },
            {
                "input":[...],
                "output":[...],
            },
            ...
        ],
    "1":[
            {
                "input":[...],
                "output":[...],
            },
            {
                "input":[...],
                "output":[...],
            },
            ...
        ],
}
```

## Use example

We provide a use example in generate.py. Generate input data of Cosmos task in Verl. Controlling the output length by ignore_eos=True and max output length = output length.

An example of generating workloads using fitted distributions is shown in the generate_distribution.py, for more stringent data desensitization. We use a Gaussian mixture distribution method to fit the distribution of model B workloads, extract the corresponding parameters to obtain the distribution, and then sample from it to generate workload data.

## Notes

In open-source tasks, some inputs cannot correspond one-to-one with outputs due to inconsistent IO during recording or checkpointing, but this does not affect data sampling.