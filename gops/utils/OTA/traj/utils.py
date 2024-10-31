from dataclasses import dataclass
import numpy as np
import pickle
import csv
import torch
from typing import Literal, NamedTuple, Union
from copy import deepcopy

# from gops.env.env_gen_ocp.resources.idsim_model.model import IdSimModel
from gops.env.env_gen_ocp.pyth_base import ContextState, State
from gops.env.env_gen_ocp.resources.idsim_model.multilane.context import MultiLaneContext
from gops.env.env_gen_ocp.resources.idsim_model.model_context import Parameter, BaseContext, State as ModelState
from gops.env.env_gen_ocp.resources.idsim_model.observe.ref import compute_onref_mask
from gops.env.env_gen_ocp.pyth_idsim import get_idsimcontext, idSimEnv, CloudServer, IdSimModel, LasvsimEnv

one_array = Union[list, np.ndarray]

"""
The Header are made by

ref x, y, phi, speed
"""

info_time = ['time_stamp_received', 'idc_time_stamp', 'time']
info_ego  = [ 'x_abs', 'y_abs', 'x', 'y', 'phi', 'vx', 'vy', 'yaw_rate', 
             'ref_index', 'acc_x', 'angle_front_wheel', 'angle_steer_wheel', 'real_acc_x', 'real_angle_front_wheel', 
             'real_angle_steer_wheel', ]
info_condi = ['controller_time(ms)', 'safety', 'comfort', 'efficiency', 'perception_t', 'drive_mode']

info_ref = ["x", "y", "phi", "speed"]
info_obs = ["x", "y", "phi", "vx", "vy"]
info_obs_head = ['type', 'length', 'width', 'idint', 'idstring']

# index indicates

n_sur = 35 # 0-34
l_sur = 61 # 0-60 2*30+1
n_ref = 61 # 0-60 2*30+1

l_time = len(info_time)
s_time = 0
l_ego = len(info_ego)
s_ego = s_time + l_time
l_condi = len(info_condi)
s_condi = s_ego + l_ego

l_basic = l_time + l_ego + l_condi
s_ref = s_condi + l_condi
cl_ref = len(info_ref)

s_sur = s_ref + n_ref*cl_ref
(cl_sur)= len(info_obs)
cl_sur_h = len(info_obs_head)

l_data = l_basic + n_ref*cl_ref + n_sur * n_ref * len(info_obs)

# Components

# self._state = np.array([x, y, u, v, phi, w]) [2,3,5]
comp_ego_state = ["x", "y", "vx", "vy", "phi", "yaw_rate"] # x, y, vx, vy, phi, r = ego_state
comp_action = ['acc_x', 'angle_steer_wheel'] # ['acc_x', 'angle_front_wheel', 'angle_steer_wheel']
comp_real_action = ['real_acc_x', 'real_angle_steer_wheel'] # ['real_acc_x', 'real_angle_front_wheel', 'real_angle_steer_wheel']

def coordinate_transformation(x_0, y_0, phi_0, x, y, phi):
    x = x.astype(float)
    y = y.astype(float)
    x_0 = float(x_0)
    y_0 = float(y_0)
    phi_0 = float(phi_0)
    phi = phi.astype(float)
    x_ = (x.astype(float) - float(x_0)) * np.cos(float(phi_0)) + (y.astype(float) - float(y_0)) * np.sin(float(phi_0))
    y_ = -(x - x_0) * np.sin(phi_0) + (y - y_0) * np.cos(phi_0)
    phi_ = phi - phi_0
    return np.stack((x_, y_, phi_), axis=-1)

def sur2constraint(ego_state, sur_states):

    return 


def parse_sur(sur_states:list, raw=False):
    """
    From sur with future to sur with only current
    """
    _s = s_sur if raw else 0
    ret = []
    for i in range(n_sur):
        start = _s+i*(cl_sur*l_sur + cl_sur_h)
        end = start + cl_sur_h + cl_sur
        ret += sur_states[start:end]
    return ret

@dataclass
class Traj:
    ego_state: one_array
    ref_state: one_array
    sur_state: one_array
    last_last_action: one_array
    last_action: one_array
    action: one_array
    action_real: one_array
    nominal_acc: one_array
    nominal_steer: one_array
    onref_mask: one_array
    
    @classmethod
    def parse_from_list(cls, data:list, l_act:list, ll_act:list, env:idSimEnv, **kwargs):
        _ego = dict(zip(info_ego, data[s_ego: s_ego+l_ego]))
        ego_state = [_ego[i] for i in comp_ego_state]
        ref_state = data[s_ref:s_ref+n_ref*cl_ref]
        sur_state = data[s_sur:s_sur+n_sur*(l_sur*(cl_sur)+cl_sur_h)]
        last_last_action = ll_act
        last_action = l_act
        action = [_ego[i] for i in comp_action]
        action_real = [_ego[i] for i in comp_real_action]
        cur_state = State(ego_state + l_act + l_act, ContextState(ref_state, sur_state, 0))
        context = get_idsimcontext(
                    cur_state, 
                    mode="batch", 
                    scenario=env.scenario
                )
        nominal_acc = env.server.model._get_nominal_acc_by_state
        nominal_steer = env.server.model._get_nominal_acc_by_state
        onref_mask = compute_onref_mask(BaseContext(context))
        return cls(ego_state, ref_state, sur_state, 
                   last_last_action, last_action, action, 
                   action_real, nominal_acc, nominal_steer, 
                   onref_mask)

    def get_reward(self, env:LasvsimEnv):

        context = get_idsimcontext(
            State.stack([env._state]), 
            mode="batch", 
            scenario=env.scenario
        )
        context.x = self.ego_state
        model_free_reward = env.model_free_reward(context, self.last_last_action, self.last_action, self.action)
        # Currently considering the model free reward is enough, for the scenario based reward, will inplemented later
        return model_free_reward

    def get_ego_obs(self):
        return self.ego_state[[2,3,5]].reshape(-1)

    def get_ref_obs(self):
        ref_info = self.ref_state.reshape((-1,cl_ref))
        ego_info = self.ego_state.reshape((1,-1))
        ref_obs = np.concatenate(
            (
                coordinate_transformation(
                    ego_info[:, 0], ego_info[:, 1], ego_info[:, 4], ref_info[:, 0], ref_info[:, 1], ref_info[:, 2]
                ), 
                ref_info[:, 3].reshape(-1, 1)
            ), axis=1
        ).reshape(-1) 
        return ref_obs

    def get_sur_obs(self):
        neighbor_info = self.sur_state.reshape(-1,(cl_sur))
        ego_info = self.ego_state.reshape((1,-1))
        state_egotrans = coordinate_transformation(
                    ego_info[:, 0], ego_info[:, 1], ego_info[:, 4], neighbor_info[:, 0], neighbor_info[:, 1], neighbor_info[:, 2]
        )
        neighbor_obs = np.concatenate(
            (state_egotrans[:, :2], np.cos(state_egotrans[:, 2:3]), np.sin(state_egotrans[:, 2:3]), neighbor_info[:, 3:]), axis=1
        ).reshape(-1)
        return neighbor_obs
    
    def get_obs(self):
        ego_obs = self.get_ego_obs()
        ref_obs = self.get_ref_obs()
        neighbor_obs = self.get_sur_obs()
        obs = np.concatenate((ego_obs, ref_obs, neighbor_obs))
        obs = obs.astype(np.float32)
        return obs
    pass


def parse_csv_to_trajectory(file_path):
    trajectories = {}
    
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            cols = reader.__next__()
            ncols = len(cols)
            sur_state = cols[s_sur:s_sur+n_sur*(l_sur*(cl_sur)+cl_sur_h)]
            ret =  parse_sur(sur_states=sur_state)
            # for row in reader:
            #     if not row or len(row) < ncols:
            #         break

            #     break

    except (FileNotFoundError, IOError) as e:
        print(f"Error opening file: {e}")
    
    return trajectories


if __name__ == "__main__":
    # Example usage
    file_path = '/home/zhengziang/code/gops-qx/OTA_server/data/idc_controler_2024-8-21_13-56-8_default.csv'
    trajectory_data = parse_csv_to_trajectory(file_path)
    pass