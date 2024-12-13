import numpy as np
import torch

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext, State, Parameter
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig, ego_mean_array, ego_std_array, ego_lower_bound_array, ego_upper_bound_array

# from gops.env.env_gen_ocp.resources.idsim_model.multilane.ref import get_ref_param, update_ref_param
from gops.env.env_gen_ocp.resources.idsim_model.multilane.sur import get_sur_param,predict_sur
from gops.env.env_gen_ocp.resources.idsim_model.multilane.traffic import get_traffic_light_param
from gops.env.env_gen_ocp.resources.idsim_model.lasvsim_env_qianxing import LasvsimEnv


class MultiLaneContext(BaseContext):
    def advance(self, x: State) -> "MultiLaneContext":
        return MultiLaneContext(x=x, p=self.p, t=self.t, i=self.i + 1,)
    @classmethod
    def from_env(cls, env: LasvsimEnv, model_config: ModelConfig, ref_index_param: int = None) -> "MultiLaneContext":
        rng = env.np_random  # random number generator
        ego_state = env._state  #x,y,u,v,phi,w
        last_last_action = env.last_action  # real value
        last_action = env.action  # real value
        sumo_time = env.timestamp
        ref_param_origin = env._ref_points   # 21,4  (x,y,phi,u)
        ref_param = np.tile(ref_param_origin, (3, 1, 1))  # 在新的维度上复制 3 次 (3,21,4)
        sur_state = env._neighbor_state  # 5,7  (x,y,phi,u,l,w,mask)
        sur_param =  predict_sur(sur_state,model_config)

        left_bound = -1
        right_bound = -1
        boundary_param = np.array([left_bound, right_bound], dtype=np.float32)
        light_param = np.array([0.]) #TODO: modify this
        # ref_param = update_ref_param(env, ref_param, light_param, model_config)
        if ref_index_param is None:
            ref_index_param = np.array(0)
        else:
            ref_index_param = np.array(ref_index_param)
        ## TODO:shape not match
        # # add noise to ego_state
        # ego_noise = rng.normal(
        #     ego_mean_array, ego_std_array, size=ego_state.shape)
        # ego_noise = np.clip(ego_noise, ego_lower_bound_array,
        #                     ego_upper_bound_array)
        # # add noise only for vx >= 1m/s
        # ego_state += ego_noise * (ego_state[2] >= 1)
        # numpy to tensor
        ego_state = torch.from_numpy(ego_state).float()
        last_last_action = torch.from_numpy(last_last_action).float()
        last_action = torch.from_numpy(last_action).float()
        ref_param = torch.from_numpy(ref_param).float()
        sur_param = torch.from_numpy(sur_param).float()
        light_param = torch.from_numpy(light_param).float()
        ref_index_param = torch.from_numpy(ref_index_param).long()
        boundary_param = torch.from_numpy(boundary_param).float()

        return MultiLaneContext(
            x=State(ego_state=ego_state,
                    last_last_action=last_last_action, last_action=last_action),
            p=Parameter(ref_param=ref_param, sur_param=sur_param,
                        light_param=light_param, ref_index_param=ref_index_param, boundary_param=boundary_param
                        ),
            t=sumo_time,
            i=0,
        )
