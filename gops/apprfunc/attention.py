__all__ = [
    "AttentionPolicy",
    "AttentionFullPolicy",
    "AttentionStateValue",
]

import torch
import itertools
import torch.nn as nn
from gops.utils.common_utils import get_activation_func, FreezeParameters
from gops.apprfunc.mlp import mlp
from typing import Tuple
from gops.apprfunc.base_attention import BaseAttention
from gops.utils.act_distribution_cls import Action_Distribution

class AttentionPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.attn = BaseAttention(**kwargs)
        self.attn_freeze = kwargs.get("attn_freeze", "none") == "policy"
        #obs_dim = kwargs["obs_dim"]+1
        obs_dim = kwargs["obs_dim"]-( kwargs["attn_end"]-kwargs["attn_begin"]+1)+kwargs["attn_out_dim"]+1
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

    def shared_params(self):
        return self.attn.parameters()
    
    def ego_params(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.attn])
    
    def forward(self, obs, virtual_t:int=1):
        with FreezeParameters([self.attn], self.attn_freeze):
            obs_processed = self.attn.forward(obs)
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs_processed, virtual_t), 1)
        actions = self.pi(expand_obs).reshape(obs.shape[0], self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action
    
    
class AttentionFullPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.attn = BaseAttention(**kwargs)
        self.attn_freeze = kwargs.get("attn_freeze", "none") == "policy"
        #obs_dim = kwargs["obs_dim"]+1
        obs_dim = kwargs["obs_dim"]-( kwargs["attn_end"]-kwargs["attn_begin"]+1)+kwargs["attn_out_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

    def shared_params(self):
        return self.attn.parameters()
    
    def ego_params(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.attn])
    
    def forward_all_policy(self, obs):
        with FreezeParameters([self.attn], self.attn_freeze):
            obs_processed = self.attn.forward(obs)
        actions = self.pi(obs_processed).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action

    def forward(self, obs):
        return self.forward_all_policy(obs)[0, :]

class AttentionStateValue(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        self.pre_horizon = kwargs["pre_horizon"]
        self.attn = BaseAttention(**kwargs)
        self.attn_freeze = kwargs.get("attn_freeze", "none") == "value"
        #obs_dim = kwargs["obs_dim"]+1
        obs_dim = kwargs["obs_dim"]-( kwargs["attn_end"]-kwargs["attn_begin"]+1)+kwargs["attn_out_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        v_sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.v = mlp(
            v_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def shared_params(self):
        return self.attn.parameters()
        
    def ego_params(self):
        return itertools.chain(*[modules.parameters() for modules in self.children() if modules != self.attn])
    
    def forward(self, obs):
        with FreezeParameters([self.attn], self.attn_freeze):
            obs_processed = self.attn.forward(obs)
        v = self.v(obs_processed)
        return torch.squeeze(v, -1)