#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Attention network
#  Update: 2023-07-03, Tong Liu: create attention
__all__ = [
    "BaseAttention",
]
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from abc import abstractmethod, ABCMeta
from gops.utils.act_distribution_cls import Action_Distribution
from gops.utils.common_utils import get_activation_func
from gops.apprfunc.mlp import mlp



def init_weights(m):
    if isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

class BaseAttention(Action_Distribution, nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.begin = kwargs["attn_begin"]
        self.end = kwargs["attn_end"]
        self.d_others = self.end - self.begin + 1
        self.d_obj = kwargs["attn_in_per_dim"]
        self.d_model = kwargs["attn_out_dim"]
        assert self.d_others % self.d_obj == 0
        self.num_objs = int(self.d_others / self.d_obj)
        print("Attention num_objs:", self.num_objs)

        # obs_dim = kwargs["obs_dim"]
        # self.act_dim = kwargs["act_dim"]
        # hidden_sizes = kwargs["hidden_sizes"]
        # self.pre_horizon = self.pre_horizon if isinstance(self.pre_horizon, int) else 1
        # pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]
        # self.pi = mlp(
        #     pi_sizes,
        #     get_activation_func(kwargs["hidden_activation"]),
        #     get_activation_func(kwargs["output_activation"]),
        # )

        self.embedding = nn.Sequential(
            nn.Linear(self.d_obj - 1, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )
        self.Uq = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)
        self.Ur = nn.Linear(self.d_model, self.d_model, bias=False, dtype=torch.float32)

        init_weights(self.embedding)
        init_weights(self.Uq)
        init_weights(self.Ur)

    # @abstractmethod
    # def preprocessing(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    #     # Return attention_obs("torch.Tensor") for attention_forward() and all other necessary auxiliary data("Tuple") in a tuple.
    #     ...
        
    def preprocessing(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        attention_obs = obs[:, self.begin:(self.end+1)]
        auxiliary = torch.concat([obs[:, 0:self.begin], obs[:, (self.end+1):]], dim = -1)
        return attention_obs, auxiliary
    
    def postprocessing(self, attention_obs: torch.Tensor, auxiliary: Tuple) -> torch.Tensor:
        obs = torch.concat([auxiliary, attention_obs], dim = -1)
        return obs
    
    def attention_forward(self, attention_obs: torch.Tensor) -> torch.Tensor:
        attention_obs = torch.reshape(attention_obs, [-1, self.num_objs, self.d_obj]) # [B, N, d_obj]
        attention_mask = attention_obs[:, :, -1].squeeze(axis=-1) # [B, N]
        attention_obs = attention_obs[:, :, :-1] # [B, N, d_obj-1]
        attention_obs = self.embedding(attention_obs)  # [B, N, d_model]

        x_real = attention_obs * attention_mask.unsqueeze(axis=-1)  # fake tensors are all zeros, [B, N, d_model]
        query = x_real.sum(axis=-2) / (attention_mask.sum(axis=-1) + 1e-5).unsqueeze(
            axis=-1)  # [B, d_model] / [B, 1] --> [B, d_model]

        logits = torch.bmm(self.Uq(query).unsqueeze(-2), self.Ur(x_real).transpose(-1, -2)).squeeze(-2)  # [B, 1, d_model] * [B, d_model, N] --> [B, 1, N] --> [B, N]

        logits = logits + ((1 - attention_mask) * -1e9)
        attention_weights = torch.softmax(logits, axis=-1) # [B, N]

        # 1) attention_obs is the weighted sum of x_real, where the weights are attention_weights
        attention_obs = torch.matmul(attention_weights.unsqueeze(axis=-2),x_real).squeeze(axis=-2) # [B, 1, N] * [B, N, d_model] --> [B, 1, d_model] --> [B, d_model]
        
        # 2) attention_obs is the max-pooled x_real without considering the fake tensors indicated by attention_mask
        # attention_obs = torch.max(x_real + (1 - attention_mask.unsqueeze(axis=-1) * -1e9), dim=-2)[0]

        # # 3) attention_obs is the mean-pooled x_real without considering the fake tensors indicated by attention_mask
        # attention_obs = query
        return attention_obs
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        attention_obs, auxilary = self.preprocessing(obs)
        attention_obs = self.attention_forward(attention_obs)
        obs_processed = self.postprocessing(attention_obs, auxilary)
        return obs_processed

    # def postprocessing(self, attention_obs: torch.Tensor, auxiliary: Tuple) -> torch.Tensor:
    #     obs = torch.concat([auxiliary, attention_obs], dim = -1)
    #     actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
    #     action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
    #              + (self.act_high_lim + self.act_low_lim) / 2
    #     return action

    # def forward_all_policy(self, obs):
    #     attention_obs, auxilary = self.preprocessing(obs)
    #     attention_obs = self.attention_forward(attention_obs)
    #     action = self.postprocessing(attention_obs, auxilary)
    #     return action
