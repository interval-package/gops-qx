import torch
import torch.nn as nn
import pickle
import argparse
import os

# Simplified network helper classes
class InferenceHelperPolicyDSAC(nn.Module):
    def __init__(self, model, act_scale_factor, obs_scale_factor, bias, obs_config):
        super().__init__()
        self.model = model
        self.act_scale_factor = act_scale_factor
        self.obs_scale_factor = obs_scale_factor
        self.bias = bias
        self.obs_config = obs_config

    def forward(self, obs: torch.Tensor):
        obs = obs * self.obs_scale_factor
        logits = self.model.policy(obs)
        action_distribution = self.model.create_action_distributions(logits)
        action = action_distribution.mode().float()
        real_act = action * self.act_scale_factor + self.bias
        return real_act

def onnx_dsac(pkl_file, output_onnx_file, input_dim, act_scale_factor, obs_scale_factor, bias, obs_config):
    # Load the saved model from a .pkl file
    with open(pkl_file, 'rb') as f:
        model = pickle.load(f)
    
    # Wrap the model with the inference helper class
    onnx_model = InferenceHelperPolicyDSAC(model, act_scale_factor, obs_scale_factor, bias, obs_config)
    
    # Create a dummy input tensor matching the input dimensions
    example_input = torch.rand(1, input_dim)
    
    # Export to ONNX
    torch.onnx.export(
        onnx_model,             # The model
        example_input,          # The input example
        output_onnx_file,       # Output file path
        input_names=['input'],  # Input tensor name
        output_names=['output'], # Output tensor name
        opset_version=11        # ONNX opset version
    )
    # print(f"Model exported to {output_onnx_file}")

# Example usage:
if __name__ == "__main__":
    # Arguments for configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the .pkl file")
    parser.add_argument("--output_onnx_file", type=str, required=True, help="Path for the output ONNX file")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension size")
    parser.add_argument("--act_scale_factor", type=float, default=1.0, help="Action scaling factor")
    parser.add_argument("--obs_scale_factor", type=float, default=1.0, help="Observation scaling factor")
    parser.add_argument("--bias", type=float, default=0.0, help="Bias for scaling action")
    args = parser.parse_args()

    # Load and export model
    onnx_dsac(
        args.pkl_file,
        args.output_onnx_file,
        args.input_dim,
        args.act_scale_factor,
        args.obs_scale_factor,
        args.bias,
        obs_config={
            "ego_dim": 6,         # Example observation config
            "sur_dim": 10,
            "ref_dim": 3,
            "num_objs": 4,
            "num_ref_points": 2
        }
    )
