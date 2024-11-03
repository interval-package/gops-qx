#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Transform pkl network to onnx version
#  Update: 2023-01-05, Jiaxin Gao: Create codes

import contextlib
import importlib
import torch, torch.nn as nn
import onnxruntime as ort
import argparse
import os
import sys
from gops.utils.common_utils import get_args_from_json
import numpy as np
import gops.algorithm

py_file_path = os.path.abspath(__file__)
print(py_file_path)
utils_path = os.path.dirname(py_file_path)
print(utils_path)
gops_path = os.path.dirname(utils_path)
gops_path = os.path.join(gops_path, 'gops-grpc')
print('gops_path', gops_path)
# Add algorithm file to sys path
alg_file = "gops/algorithm"
alg_path = os.path.join(gops_path, alg_file)
print(alg_path)
sys.path.append(alg_path)


##########################DSACT network pkl2onnx#################

def _load_args(log_policy_dir):
    log_policy_dir = log_policy_dir
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def preprocess_obs(obs, obs_config):
    sur_start = obs_config["ego_dim"] + obs_config["num_ref_points"] * obs_config["ref_dim"]
    sur_end = sur_start + obs_config["num_objs"] * obs_config["sur_dim"]
    num_objs = obs_config["num_objs"]
    sur_dim = obs_config["sur_dim"]
    sur_objs = obs[:, sur_start:sur_end].reshape(-1, num_objs, sur_dim).clone()    
    obs[:, sur_start:sur_end] = sur_objs.reshape(-1, num_objs * sur_dim)

    return obs

@contextlib.contextmanager
def _module_inference(module: nn.Module):
    training = module.training
    module.train(False)
    yield
    module.train(training)

class _InferenceHelper_Policy_DSAC(nn.Module):
    def __init__(self, model, act_scale_factor,obs_scale_factor, bias, obs_config):
        super().__init__()
        self.model = model
        self.act_scale_factor = act_scale_factor
        self.obs_scale_factor = obs_scale_factor
        self.bias = bias
        self.obs_config = obs_config
        self.ego_dim = self.obs_config["ego_dim"]
        self.sur_dim = self.obs_config["sur_dim"]
        self.ref_dim = self.obs_config["ref_dim"]
        self.num_objs = self.obs_config["num_objs"]
        self.num_ref_points = self.obs_config["num_ref_points"]
        self.sur_start = self.ego_dim + self.num_ref_points * self.ref_dim
        self.sur_end = self.sur_start + self.num_objs * self.sur_dim


    def forward(self, obs: torch.Tensor):
        obs = preprocess_obs(obs,self.obs_config)
        obs = obs*self.obs_scale_factor
        logits = self.model.policy(obs)
        action_distribution = self.model.create_action_distributions(logits)
        action = action_distribution.mode().float()
        real_act = action*self.act_scale_factor + self.bias
        real_act = action
        return real_act
    
class _InferenceHelper_Q_DSAC(nn.Module):
    def __init__(self, model, act_dim, act_scale_factor,obs_scale_factor, bias):
        super().__init__()
        self.model = model
        self.act_dim = act_dim
        self.act_scale_factor = act_scale_factor
        self.obs_scale_factor = obs_scale_factor
        self.bias = bias



    def forward(self, obs_act: torch.Tensor):
        obs  = obs_act[:,0:-act_dim]
        obs = obs*self.obs_scale_factor
        act = obs_act[:,-act_dim:]
        act = act*self.act_scale_factor + self.bias
        logits = self.model(obs,act)
        mean ,_ = torch.chunk(logits,2,-1)
        return mean
    
def DSAC_policy_export_onnx_model(networks, input_dim, policy_dir, act_scale_factor,obs_scale_factor, bias,obs_config):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Policy_DSAC(networks, act_scale_factor,obs_scale_factor,bias,obs_config)
    torch.onnx.export(model, example, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)

def DSAC_Q1_export_onnx_model(networks, input_dim_obs, input_dim_act,policy_dir,act_scale_factor,obs_scale_factor, bias):

    example_obs_act = torch.rand(1, input_dim_obs+input_dim_act)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Q_DSAC(networks.q1,input_dim_act, act_scale_factor,obs_scale_factor,bias)
    torch.onnx.export(model, example_obs_act, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)

def DSAC_Q2_export_onnx_model(networks, input_dim_obs, input_dim_act,policy_dir,act_scale_factor,obs_scale_factor, bias):

    example_obs_act = torch.rand(1, input_dim_obs+input_dim_act)  # network input dim
    output_onnx_model = policy_dir
    model = _InferenceHelper_Q_DSAC(networks.q2,input_dim_act, act_scale_factor,obs_scale_factor,bias)
    torch.onnx.export(model, example_obs_act, output_onnx_model, input_names=['input'], output_names=['output'],
                          opset_version=11)

def main():
    # Load trained policy
    log_policy_dir = "/home/zhengziang/code/gops-qx/results/pyth_idsim/DSACTPI_241013-023014"
    index = 1000
    args = _load_args(log_policy_dir)
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = importlib.import_module("gops.algorithm." + alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir+"/apprfunc/apprfunc_{}.pkl".format(f'{index}')  # network position
    networks.load_state_dict(torch.load(log_path, weights_only=True))
    networks.eval()

    # DSAC
    obs_dim = args["obsv_dim"]
    act_dim = args["action_dim"]
    ego_dim = args["env_model_config"]["ego_feat_dim"]
    sur_dim = args["env_model_config"]["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = args["env_model_config"]["per_ref_feat_dim"]

    num_ref_points = len(args["env_model_config"]["downsample_ref_point_index"])
    num_objs = int(sum(i for i in args["env_config"]["obs_num_surrounding_vehicles"].values()))
    obs_dict = {
        "ego_dim": ego_dim,
        "sur_dim": sur_dim,
        "ref_dim": ref_dim,
        "num_objs": num_objs,
        "num_ref_points": num_ref_points,
    }
    policy_dir = os.path.join(log_policy_dir, 'dsac_policy3.onnx')
    Q1_dir = os.path.join(log_policy_dir, 'dsac_q1.onnx')
    Q2_dir = os.path.join(log_policy_dir, 'dsac_q2.onnx')
    action_upper_bound = args["env_config"]["action_upper_bound"]
    action_lower_bound = args["env_config"]["action_lower_bound"]    
    action_scale_factor = 1
    action_scale_bias = 0

    obs_scale_factor = args["obs_scale"]
    obs_scale_factor = torch.tensor(obs_scale_factor).float()
    DSAC_policy_export_onnx_model(networks, obs_dim, policy_dir,action_scale_factor,obs_scale_factor,action_scale_bias,obs_dict)
    DSAC_Q1_export_onnx_model(networks, obs_dim, act_dim, Q1_dir,action_scale_factor,obs_scale_factor,action_scale_bias)
    DSAC_Q2_export_onnx_model(networks, obs_dim, act_dim, Q2_dir,action_scale_factor,obs_scale_factor,action_scale_bias)


    # ### example of DSAC algorithm
    ort_session_policy = ort.InferenceSession(policy_dir)
    example_policy = np.random.randn(1,obs_dim).astype(np.float32)

    inputs_policy = {ort_session_policy.get_inputs()[0].name: example_policy.copy(),}
    outputs_policy = ort_session_policy.run(None, inputs_policy)
    print(outputs_policy[0])
    example_policy = preprocess_obs(torch.tensor(example_policy).clone(),obs_dict)*obs_scale_factor
    logits = networks.policy(example_policy)
    action,_ = torch.chunk(logits,2,-1) 
    action = torch.tanh(action)
    action = action*action_scale_factor + action_scale_bias
    print(action)

    ort_session_value = ort.InferenceSession(Q1_dir)
    example_obs_act = np.random.randn(1, obs_dim+act_dim).astype(np.float32)
    inputs_value = {ort_session_value.get_inputs()[0].name: example_obs_act,} 
    outputs_value = ort_session_value.run(None, inputs_value)
    print(outputs_value)
    value = networks.q1(torch.tensor(example_obs_act)[:,:-act_dim]*obs_scale_factor,torch.tensor(example_obs_act)[:,-act_dim:])
    print(value)

def onnx_brute_force(mdl_path, cfgs, output_dir):
    alg_name = cfgs["algorithm"]
    alg_file_name = alg_name.lower()
    file = importlib.import_module("gops.algorithm." + alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**cfgs)
    networks.load_state_dict(torch.load(mdl_path, weights_only=True))
    networks.eval()

    # DSAC
    obs_dim = cfgs["obsv_dim"]
    act_dim = cfgs["action_dim"]
    ego_dim = cfgs["env_model_config"]["ego_feat_dim"]
    sur_dim = cfgs["env_model_config"]["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = cfgs["env_model_config"]["per_ref_feat_dim"]

    num_ref_points = len(cfgs["env_model_config"]["downsample_ref_point_index"])
    num_objs = int(sum(i for i in cfgs["env_config"]["obs_num_surrounding_vehicles"].values()))
    obs_dict = {
        "ego_dim": ego_dim,
        "sur_dim": sur_dim,
        "ref_dim": ref_dim,
        "num_objs": num_objs,
        "num_ref_points": num_ref_points,
    }
    action_scale_factor = 1
    action_scale_bias = 0

    obs_scale_factor = cfgs["obs_scale"]
    obs_scale_factor = torch.tensor(obs_scale_factor).float()
    DSAC_policy_export_onnx_model(networks, obs_dim, output_dir,action_scale_factor,obs_scale_factor,action_scale_bias,obs_dict)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    # main()
    # log_policy_dir = "/home/zhengziang/code/gops-qx/results/pyth_idsim/DSACTPI_241013-023014"
    # source = "results/pyth_idsim/DSACTPI_241013-023014/apprfunc/apprfunc_1000.pkl"
    # target = "results/pyth_idsim/DSACTPI_241013-023014/onnx/1.0.0.0.onnx"
    log_policy_dir = args.ckpt
    source = args.source
    target = args.target

    json_path = os.path.join(log_policy_dir, "config.json")
    import json

    summary_filename = json_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    args = summary_dict

    onnx_brute_force(source, args, target)
