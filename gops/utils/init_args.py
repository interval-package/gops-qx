#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: initialize parameters
#  Update: 2021-03-10, Yuhang Zhang: Revise Codes


import copy
import datetime
import json
import os
import ray
import torch
import warnings
from gym.spaces import Box, Discrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gops.utils.common_utils import change_type, seed_everything



def init_args(env, **args):
    # set torch parallel threads nums in main process
    num_threads_main = args.get("num_threads_main", None)
    if num_threads_main is None:
        if "serial" in args["trainer"]:
            num_threads_main = 1
        else:
            num_threads_main = 1
    torch.set_num_threads(num_threads_main)
    print("limit torch intra-op parallel threads num in main process "
          "to {num} for saving computing resource.".format(num=num_threads_main))

    # cuda
    if args["enable_cuda"]:
        if torch.cuda.is_available():
            args["use_gpu"] = True
        else:
            warning_msg = "cuda is not available, use CPU instead"
            warnings.warn(warning_msg)
            args["use_gpu"] = False
    else:
        args["use_gpu"] = False

    # sampler
    if args["trainer"] == "on_sync_trainer":
        args["batch_size_per_sampler"] = (
            args["sample_batch_size"] // args["num_samplers"]
        )
        if args["sample_batch_size"] % args["num_samplers"] != 0:
            args["sample_batch_size"] = (
                args["batch_size_per_sampler"] * args["num_samplers"]
            )
            error_msg = (
                "sample_batch_size can not be exact divided by the number of samplers!"
            )
            raise ValueError(error_msg)
    else:
        args["batch_size_per_sampler"] = args["sample_batch_size"]

    # observation dimension
    if len(env.observation_space.shape) == 1:
        args["obsv_dim"] = env.observation_space.shape[0]
    else:
        args["obsv_dim"] = env.observation_space.shape

    if isinstance(env.action_space, (Box, GymnasiumBox)):
        # get dimension of continuous action or num of discrete action
        args["action_type"] = "continu"
        args["action_dim"] = (
            env.action_space.shape[0]
            if len(env.action_space.shape) == 1
            else env.action_space.shape
        )
        args["action_high_limit"] = env.action_space.high.astype("float32")
        args["action_low_limit"] = env.action_space.low.astype("float32")
    elif isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        args["action_type"] = "discret"
        args["action_dim"] = 1
        args["action_num"] = env.action_space.n
        args["noise_params"]["action_num"] = args["action_num"]

    if hasattr(env, "constraint_dim"):
        args["constraint_dim"] = env.constraint_dim

    if "value_func_type" in args.keys() and args["value_func_type"] == "CNN_SHARED":
        if hasattr(args, "policy_func_type"):
            assert (
                args["value_func_type"] == args["policy_func_type"]
            ), "The function type of both value and policy should be CNN_SHARED"
            assert (
                args["value_conv_type"] == args["policy_conv_type"]
            ), "The conv type of value and policy should be the same"
        args["cnn_shared"] = True
        args["feature_func_name"] = "Feature"
        args["feature_func_type"] = "CNN_SHARED"
        args["conv_type"] = args["value_conv_type"]
    else:
        args["cnn_shared"] = False

    if "value_func_type" in args.keys() and args["value_func_type"] == "PINet":
        if "policy_func_type" in args.keys():
            assert (
                args["value_func_type"] == args["policy_func_type"]
            ), "The function type of both value and policy should be PINet"
        args["PI_shared"] = True
        args["pi_net_func_name"] = "PINet"
        args["pi_net_func_type"] = "PINet"
    else:
        args["PI_shared"] = False

    # Create save arguments
    if args["save_folder"] is None:
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.dirname(dir_path)
        dir_path = os.path.dirname(dir_path)
        args["save_folder"] = os.path.join(
            dir_path + "/results/",args["env_id"],
            args["algorithm"] +'_'+
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
    os.makedirs(args["save_folder"], exist_ok=True)
    os.makedirs(args["save_folder"] + "/apprfunc", exist_ok=True)
    os.makedirs(args["save_folder"] + "/evaluator", exist_ok=True)

    # set random seed
    seed = args.get("seed", None)
    args["seed"] = seed_everything(seed)
    print("Set global seed to {}".format(args["seed"]))
    with open(args["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    args["additional_info"] = env.additional_info
    for key, value in args["additional_info"].items():
        args[key+"_dim"] = value["shape"]

    # Start a new local Ray instance
    # This is necessary since all training scripts use evaluator, which uses ray.
    ray.init(address="local")

    return args
