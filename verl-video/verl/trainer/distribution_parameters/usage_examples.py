# Task Data Generator Usage Examples
# Generated on 2025-08-21T15:06:29.919702

import json
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# Load parameters
with open('task_distribution_parameters.json', 'r') as f:
    params = json.load(f)


# Example 1: Generate math task data
generator = TaskDataGenerator(exported_params)
math_data = generator.generate_math_data(n_samples=1000)
print(f"Generated {len(math_data['input_len'])} math samples")

# Example 2: Generate tool calling data for specific call count
tool_data_3_calls = generator.generate_tool_calling_data(call_count=3, n_samples=100)
print(f"Generated {len(tool_data_3_calls)} 3-call sequences")

# Example 3: Generate mixed tool calling data
mixed_data = generator.generate_mixed_tool_calling_data(n_samples=500)
print(f"Generated {len(mixed_data)} mixed tool calling sequences")

# Example 4: Check generation capabilities
summary = generator.get_generation_summary()
print("Available capabilities:", summary)

