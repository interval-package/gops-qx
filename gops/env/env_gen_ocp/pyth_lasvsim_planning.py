import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Generic, Optional, Tuple, Union
from typing_extensions import Self
from copy import deepcopy

import gym
import time
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from gops.env.env_gen_ocp.pyth_idsim import idSimEnv, get_idsimcontext
from gops.env.env_gen_ocp.resources.idsim_var_type import Config
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig

# from idsim.config import Config
# from idsim.envs.env import CrossRoad
# from idsim_model.model import IdSimModel
# from idsim_model.model_context import Parameter, BaseContext
# from idsim_model.crossroad.context import CrossRoadContext
# from idsim_model.multilane.context import MultiLaneContext
# from idsim_model.model_context import State as ModelState
# from idsim_model.params import ModelConfig


class TrajectoryProcessor:
    def __init__(self, dense_ref_mode: str, dense_ref_param: Optional[Any] = None):
        dense_func_name = f"dense_ref_{dense_ref_mode}"
        dense_ref = getattr(self, dense_func_name)
        self.dense_ref = partial(dense_ref, dense_ref_param=dense_ref_param)

        self.traj_vocabulary = None
        if dense_ref_mode == "vocabulary":
            self.traj_vocabulary = self.get_vocabulary(traj_path = dense_ref_param)
        
    def get_vocabulary(self, traj_path: str) -> np.ndarray:
        """
        get vocabulary from trajectory file.

        Parameters:
        traj_path (str): path of *.npy trajectory file in ego vehicle frame
                                

        Returns:
        np.ndarray: traj_vocabulary in ego vehicle frame
        """
        dt = 0.1 # FIXME: hard code
        ref = np.load(traj_path) # ref：[R, N, 3]
        ref_v = np.sum(np.diff(ref[:, :, :2], axis=1) ** 2, axis=-1) ** 0.5 / dt # [R, N-1]
        last_v = ref_v[:, -1]
        ref_v = np.concatenate([ref_v, last_v[:, None]], axis=1) # [R, N]
        return np.concatenate([ref, ref_v[:, :, None]], axis=-1) # [R, N, 4]
    
    def generate_bezier_curve_with_phi(self, origin_point:np.array, dest_point:np.array, n_points=100) -> np.array:
        x0, y0, phi0, v_o = origin_point
        x3, y3, phi3, v_d = dest_point
        delta_v = v_d - v_o
        p1_x = x0 + np.cos(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        p1_y = y0 + np.sin(phi0) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        
        p2_x = x3 - np.cos(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        p2_y = y3 - np.sin(phi3) * 0.4 * np.linalg.norm([x3 - x0, y3 - y0])
        
        P0 = np.array([x0, y0])
        P1 = np.array([p1_x, p1_y])
        P2 = np.array([p2_x, p2_y])
        P3 = np.array([x3, y3])

        t_values = np.linspace(0, 1, n_points)

        bezier_points = []
        for t in t_values:
            x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
            y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]

            dx = 3 * (1 - t)**2 * (P1[0] - P0[0]) + 6 * (1 - t) * t * (P2[0] - P1[0]) + 3 * t**2 * (P3[0] - P2[0])
            dy = 3 * (1 - t)**2 * (P1[1] - P0[1]) + 6 * (1 - t) * t * (P2[1] - P1[1]) + 3 * t**2 * (P3[1] - P2[1])

            phi = np.arctan2(dy, dx)
            bezier_points.append(np.array([x, y, phi, v_o + delta_v * t]))

        bezier_points = np.array(bezier_points)
        return bezier_points

    def dense_ref_bezier(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param: Optional[list] = None) -> np.ndarray:
        """
        Densify reference parameters by add Bezier curves.

        Parameters:
        ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                                Each element represents [ref_x, ref_y, ref_phi, ref_v].

        Returns:
        np.ndarray: Densified reference parameters with shape [R+(R-1)*2*len(ratio_list), 2N+1, 4].
                    Each element represents [ref_x, ref_y, ref_phi, ref_v].
        """
        ratio_list = [1] if dense_ref_param is None else dense_ref_param
        bezier_list=[]
        num_point = ref_param.shape[-2]
        for sample_ratio in ratio_list:
            target_index = int(sample_ratio * num_point)
            for i in range(ref_param.shape[0]):
                if i == 0:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                elif i==ref_param.shape[0]-1:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                else:
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i-1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i-1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                    
                    ref_bezier = self.generate_bezier_curve_with_phi(ref_param[i][0], ref_param[i+1][target_index-1],target_index)
                    if int(num_point-target_index)!=0:
                        bezier_list.append(np.concatenate((ref_bezier,ref_param[i+1][-int(num_point-target_index):])))
                    else:
                        bezier_list.append(ref_bezier)
                        
        return np.concatenate([np.array(bezier_list),ref_param])

    def dense_ref_boundary(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        """
        Densify reference parameters by add boundaries.

        Parameters:
        ref_param (np.ndarray): Input reference parameters with shape [R, 2N+1, 4].
                                Each element represents [ref_x, ref_y, ref_phi, ref_v].

        Returns:
        np.ndarray: Densified reference parameters with shape [2R-1, 2N+1, 4].
                    Each element represents [ref_x, ref_y, ref_phi, ref_v].
        """

        A, B, C = ref_param.shape
        ret = np.zeros((2*A - 1, B, C))

        for j in range(A):
            ret[2*j, :, :] = ref_param[j, :, :]
        for j in range(A - 1):
            ret[2*j + 1, :, :] = (ref_param[j, :, :] + ref_param[j + 1, :, :]) / 2

        return ret

    def dense_ref_no_dense(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        return ref_param

    def dense_ref_vocabulary(self, ref_param: np.ndarray, robot_state: np.ndarray = None, dense_ref_param = None):
        """
        transform self.traj_vocabulary from ego vehicle frame to world frame.
        xR_world = xE_world + xR_ego * cos(phiE) - yR_ego * sin(phiE)
        yR_world = yE_world + xR_ego * sin(phiE) + yR_ego * cos(phiE)
        phiR_world = phiE_world + phiR_ego
        shape of traj_vocabulary: [R, N, 4]

        Parameters:
        robot_state (np.ndarray): [x, y, vx, vy, phi, omega, last_last_action, last_action]
                                

        Returns:
        np.ndarray: traj_vocabulary in world frame
        """
        vocabulary_in_ego = self.traj_vocabulary
        ego_in_world = np.concatenate([robot_state[..., :2], robot_state[..., 4:5]], axis=-1)[None, None, :] # [1, 1, 3] x, y, phi
        vocabulary_in_world = np.zeros_like(vocabulary_in_ego)
        vocabulary_in_world[..., 0] = vocabulary_in_ego[..., 0] * np.cos(ego_in_world[..., 2]) - vocabulary_in_ego[..., 1] * np.sin(ego_in_world[..., 2]) + ego_in_world[..., 0]
        vocabulary_in_world[..., 1] = vocabulary_in_ego[..., 0] * np.sin(ego_in_world[..., 2]) + vocabulary_in_ego[..., 1] * np.cos(ego_in_world[..., 2]) + ego_in_world[..., 1]
        vocabulary_in_world[..., 2] = vocabulary_in_ego[..., 2] + ego_in_world[..., 2]
        vocabulary_in_world[..., 3] = vocabulary_in_ego[..., 3]
        return vocabulary_in_world

class idSimEnvPlanning(idSimEnv):
    def __init__(self, env_config: Config, model_config: ModelConfig, 
                 scenario: str, rou_config: Dict[str, Any]=None, env_idx: int=None, scenerios_list: List[str]=None):
        super(idSimEnvPlanning, self).__init__(env_config, model_config, scenario, rou_config, env_idx, scenerios_list)

        self.ref_vocabulary = None
        self.planning_horizon = 0
        self.cum_reward_list = None
        self.cum_critic_comps_list = None
        
        self.traj_processor = TrajectoryProcessor(dense_ref_mode=env_config.dense_ref_mode, dense_ref_param=env_config.dense_ref_param)
        print(f"INFO: dense ref mode: {env_config.dense_ref_mode}")
        print(f"INFO: dense ref param: {env_config.dense_ref_param}")

    def set_plan_beginning(self):
        self.begin_planning = True
        self.planning_horizon = 0
            
    # @cal_ave_exec_time(print_interval=1000)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, terminated, truncated, info = super(idSimEnv, self).step(action)

        # ----- set ref_index to the middle lane to calculate obs -----
        mid_index = self._state.context_state.reference.shape[0] // 2
        self._state = self._get_state_from_idsim(ref_index_param=mid_index) # get state using mid_index to calculate obs

        #  ----- initialize cumulative reward and replan ref at beginning of planning -----
        if self.begin_planning:
            self.begin_planning = False
            self.ref_vocabulary = self.traj_processor.dense_ref(ref_param = self._state.context_state.reference, robot_state = self._state.robot_state)
            self.cum_reward_list = [0.0] * self.ref_vocabulary.shape[0]
            cum_critic_comps = np.zeros(len(self.critic_dict), dtype=np.float32)
            self.cum_critic_comps_list = [copy.deepcopy(cum_critic_comps) for _ in range(self.ref_vocabulary.shape[0])]

        # ----- re-calculate ref and time horizon -----
        replan_state = copy.deepcopy(self._state)
        replan_state.context_state.t = np.array(self.planning_horizon, dtype=np.int32)
        replan_state.context_state.reference = self.ref_vocabulary
        self.planning_horizon += 1

        # ----- get reward for each reference -----        
        reward_model_free_list, mf_info_list = self._get_model_free_reward_by_state_batch(replan_state, action)

        # ----- get cumulated reward and critic components for each reference -----
        self.cum_reward_list = [cum_r + r + reward for cum_r, r in zip(self.cum_reward_list, reward_model_free_list)]
        critic_comps_list = [self.get_critic_comps({**info, **mf_info}) for mf_info in mf_info_list]
        self.cum_critic_comps_list = [cum_r + r for cum_r, r in zip(self.cum_critic_comps_list, critic_comps_list)]

        # ----- choose highest cumulated reward -----
        opt_ref_index = np.argmax(self.cum_reward_list)
        total_reward = self.cum_reward_list[opt_ref_index]
        
        # ----- update info -----
        info.update(mf_info_list[opt_ref_index])
        info["reward_details"] = {}
        done = terminated or truncated
        if truncated:
            info["TimeLimit.truncated"] = True # for gym

        self._info = self._get_info(info)
        self._info["critic_comps"] = self.cum_critic_comps_list[opt_ref_index]

        # if not terminated:
        #     total_reward = np.maximum(total_reward, 0.05)

        return self._get_obs(), total_reward, done, self._info

    def _get_model_free_reward_by_state_batch(self, state: State, action: np.ndarray) -> float:
        idsim_context = get_idsimcontext(
            State.stack([state]), 
            mode="batch", 
            scenario=self.scenario
        )

        reward, info = self.model_free_reward_batch(
            context=idsim_context,
            last_last_action=state.robot_state[..., -4:-2][None, :], # absolute action
            last_action=state.robot_state[..., -2:][None, :], # absolute action
            action=action[None, :] # incremental action
        )
        
        return reward, info


def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    assert "env_config" in kwargs.keys(), "env_config must be specified"
    env_config = deepcopy(kwargs["env_config"])

    assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
    env_scenario = kwargs["env_scenario"]

    assert 'scenario_root' in env_config, "scenario_root must be specified in env_config"
    env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)

    assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
    model_config = deepcopy(kwargs["env_model_config"])
    model_config = ModelConfig.from_partial_dict(model_config)

    env_idx = kwargs["env_idx"] if "env_idx" in kwargs.keys() else 0

    qx_config = kwargs.get("qx_config", None)

    env = idSimEnv(env_config, model_config, env_scenario, env_idx, qx_config=qx_config)
    return env