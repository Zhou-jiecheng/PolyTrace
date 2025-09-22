"""
For stricter data confidentiality requirements, you can fit a probability distribution and then generate input metadata based on the fitted distribution.
A example of Model B function fitter
"""
import pickle
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, ks_2samp
import warnings
warnings.filterwarnings('ignore')

class EnhancedTaskDataGenerator:
    """
    Generate close source RL training workloads
    """
    
    def __init__(self, parameters_file: Optional[str] = None):
        """
        Initializes the data generator.
        
        Args:
            parameters_file: Path to the pre-trained parameters file (optional).
        """
        self.math_gmm = None
        self.math_input_len = None
        self.math_output_len = None
        self.sequence_models = {}
        self.call_count_probabilities = {}
        self.is_trained = False
        
        if parameters_file:
            self.load_parameters(parameters_file)

    def generate_math_samples(self, n_samples: int = 1000) -> List[Dict]:
        if not self.is_trained or self.math_gmm is None:
            raise RuntimeError("Distribution models have not been trained.")
        
        # Sample from the Gaussian Mixture Model
        samples, _ = self.math_gmm.sample(n_samples)
        
        # Ensure positive values
        input_lengths = np.maximum(1, np.round(samples[:, 0]).astype(int))
        output_lengths = np.maximum(1, np.round(samples[:, 1]).astype(int))
        
        # Create a list of samples
        math_samples = []
        for i in range(n_samples):
            sample = {
                'task_type': 'math',
                'input_len': int(input_lengths[i]),
                'output_len': int(output_lengths[i]) if int(output_lengths[i]) < 32000 else 32000,
                'generation_method': 'gaussian_mixture_model'
            }
            math_samples.append(sample)
        
        return math_samples
    
    def generate_tool_calling_samples(self, n_samples: int = 1000) -> List[Dict]:
        if not self.is_trained or not self.sequence_models:
            raise RuntimeError("Distribution models have not been trained.")

        tool_calling_samples = []
        
        # Get available call counts and probabilities
        call_counts = list(self.call_count_probabilities.keys())
        # Only use call counts that have a trained model
        available_call_counts = [c for c in call_counts if c in self.sequence_models]
        
        # Recalculate the probability distribution
        total_prob = sum(self.call_count_probabilities[c] for c in available_call_counts)
        probabilities = [self.call_count_probabilities[c] / total_prob for c in available_call_counts]
        
        for i in range(n_samples):
            # Select the number of calls
            selected_call_count = np.random.choice(available_call_counts, p=probabilities)
            
            # Sample from the corresponding model
            model_info = self.sequence_models[selected_call_count]
            gmm_model = model_info['best_model']
            
            sample_vector, _ = gmm_model.sample(1)
            sample_vector = sample_vector[0]
            
            # Separate input and output lengths
            n_calls = selected_call_count
            input_lengths = np.maximum(1, np.round(sample_vector[:n_calls]).astype(int))
            output_lengths = np.maximum(1, np.round(sample_vector[n_calls:]).astype(int))
            output_lengths = np.clip(output_lengths, 1, 32000)  # Clip the output length
            sample = {
                'task_type': 'tool_calling',
                'call_count': selected_call_count,
                'input_len': input_lengths.tolist(),
                'output_len': output_lengths.tolist(),
                'generation_method': f'gaussian_mixture_model_{selected_call_count}_calls'
            }
            tool_calling_samples.append(sample)
        
        return tool_calling_samples

    def generate_samples(self, n_samples: int = 1000, task_type: str = "math") -> List[Dict]:
        """
        Generates mixed samples.
        
        Args:
            n_samples: Total number of samples.
            task_type: The type of task to generate samples for.
            
        Returns:
            A list of mixed samples.
        """
        if task_type == "math":
            samples = self.generate_math_samples(n_samples)
        else:
            samples = self.generate_tool_calling_samples(n_samples)

        # Merge and shuffle
        np.random.shuffle(samples)
        
        return samples

    def validate_samples(self, samples: List[Dict], original_data: Optional[Dict] = None, task_type: str = "math") -> Dict:
        """
        Validates the quality of the generated samples.
        
        Args:
            samples: The generated samples.
            original_data: The original data for comparison (optional).
            task_type: The type of task to validate.
            
        Returns:
            A dictionary with validation results.
        """
        if task_type == "math":
            math_inputs = [s['input_len'] for s in samples]
            math_outputs = [s['output_len'] for s in samples]
            
            # Compare with original data
            if original_data and 'math_input_len' in original_data:
                orig_inputs = original_data['math_input_len']
                orig_outputs = original_data['math_output_len']
                
                # KS-test
                ks_input = ks_2samp(math_inputs, orig_inputs)
                ks_output = ks_2samp(math_outputs, orig_outputs)

                print("Math Data two-sample Kolmogorov-Smirnov (input, output): ", ks_input, ks_output)
            return math_inputs, math_outputs, orig_inputs, orig_outputs

        if task_type == "tool_calling":
            tool_by_calls = {}
            for sample in samples:
                call_count = sample['call_count']
                if call_count not in tool_by_calls:
                    tool_by_calls[call_count] = []
                tool_by_calls[call_count].append(sample)
            print("Generate Tool calling keys:", tool_by_calls.keys())

            orig_data_dict = {}
            for s in original_data:
                k = len(s["input_len"])
                if k not in orig_data_dict:
                    orig_data_dict[k] = []
                orig_data_dict[k].append(s)
            print("Original tool calling keys:", orig_data_dict.keys())
            ks_input_list = []
            ks_output_list = []
            p_input_list = []
            p_output_list = []
            tool_calling_inputs = []
            tool_calling_outputs = []
            orig_tool_calling_inputs = []
            orig_tool_calling_outputs = []
            for k in orig_data_dict.keys():
                generate_data = tool_by_calls.get(k, [])
                orig_data = orig_data_dict[k]
                input_len_generate = [item for sublist in [s['input_len'] for s in generate_data] for item in sublist]
                output_len_generate = [item for sublist in [s['output_len'] for s in generate_data] for item in sublist]
                input_len_orig = [item for sublist in [s['input_len'] for s in orig_data] for item in sublist]
                output_len_orig = [item for sublist in [s['output_len'] for s in orig_data] for item in sublist]
                # Perform KS-test
                ks_input_res = ks_2samp(input_len_generate, input_len_orig)
                ks_output_res = ks_2samp(output_len_generate, output_len_orig)
                print(f"   tool calling {k} two-sample Kolmogorov-Smirnov (input, output): {ks_input_res}, {ks_output_res}")
                ks_input_list.append(ks_input_res.statistic)
                ks_output_list.append(ks_output_res.statistic)
                p_input_list.append(ks_input_res.pvalue)
                p_output_list.append(ks_output_res.pvalue)

                tool_calling_inputs.extend(input_len_generate)
                tool_calling_outputs.extend(output_len_generate)
                orig_tool_calling_inputs.extend(input_len_orig)
                orig_tool_calling_outputs.extend(output_len_orig)

            ks_input_list = np.array(ks_input_list)
            ks_output_list = np.array(ks_output_list)
            p_input_list = np.array(p_input_list)
            p_output_list = np.array(p_output_list)
            print("Tool Calling Data two-sample Kolmogorov-Smirnov (input ks, output ks, input p-values, output p-values): ", ks_input_list.mean(), ks_output_list.mean(), p_input_list.mean(), p_output_list.mean())
            return tool_calling_inputs, tool_calling_outputs, orig_tool_calling_inputs, orig_tool_calling_outputs

    def load_parameters(self, filename: str):
        """
        Loads model parameters.
        Args:
            filename: Path to the parameters file (.pkl or .json).
        """
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.math_gmm = data['math_gmm']
            self.sequence_models = data['sequence_models']
            self.call_count_probabilities = data['call_count_probabilities']
            self.is_trained = data.get('is_trained', True)
            
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                parameters = json.load(f)
            
            math_params = parameters['math_model']
            self.math_gmm = GaussianMixture(n_components=math_params['n_components'])
            self.math_gmm.means_ = np.array(math_params['gmm_means'])
            self.math_gmm.covariances_ = np.array(math_params['gmm_covariances'])
            self.math_gmm.weights_ = np.array(math_params['gmm_weights'])
            
            self.sequence_models = {}
            for call_count_str, model_params in parameters['tool_calling_models'].items():
                call_count = int(call_count_str)
                
                gmm = GaussianMixture(n_components=model_params['n_components'])
                gmm.means_ = np.array(model_params['gmm_means'])
                gmm.covariances_ = np.array(model_params['gmm_covariances'])
                gmm.weights_ = np.array(model_params['gmm_weights'])
                
                self.sequence_models[call_count] = {
                    'call_count': call_count,
                    'n_samples': model_params['n_samples'],
                    'n_features': model_params['n_features'],
                    'best_model': gmm,
                    'optimal_components': model_params['optimal_components'],
                    'bic_score': model_params['bic_score'],
                    'io_correlation': model_params['io_correlation']
                }
            
            self.call_count_probabilities = {int(k): v for k, v in parameters['call_count_probabilities'].items()}
            self.is_trained = True
        
        else:
            raise ValueError("Unsupported file format, please use .pkl or .json files")
        
        print(f"Parameters loaded: {filename}")
    
    def get_model_summary(self) -> Dict:
        """
        Gets a summary of the model information.
        
        Returns:
            A dictionary containing the model summary.
        """
        if not self.is_trained:
            return {'status': 'untrained'}
        
        summary = {
            'status': 'trained',
            'math_model': {
                'n_components': self.math_gmm.n_components if self.math_gmm else 0,
                'training_samples': len(self.math_input_len) if self.math_input_len else 0
            },
            'tool_calling_models': {
                'available_call_counts': list(self.sequence_models.keys()),
                'total_models': len(self.sequence_models),
                'call_count_distribution': self.call_count_probabilities
            }
        }
        
        return summary


def demo_usage():
    """Demonstrates usage."""
    print("=== Enhanced Task Data Generator Demo ===")
    
    # Create the generator
    generator = EnhancedTaskDataGenerator(parameters_file="./enhanced_task_distribution_parameters.json")

    math_generated_samples = generator.generate_samples(256,"math")
    tool_using_samples = generator.generate_samples(256,"tool_calling")


if __name__ == "__main__":
    demo_usage()
