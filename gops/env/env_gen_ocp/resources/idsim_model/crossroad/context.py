import numpy as np
import torch

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext, State, Parameter
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig, ego_mean_array, ego_std_array, ego_lower_bound_array, ego_upper_bound_array

from gops.env.env_gen_ocp.resources.idsim_model.crossroad.ref import get_ref_param
from gops.env.env_gen_ocp.resources.idsim_model.crossroad.sur import get_sur_param
from gops.env.env_gen_ocp.resources.idsim_model.lasvsim_env_qianxing import LasvsimEnv
from gops.env.env_gen_ocp.resources.idsim_model.crossroad.sur import get_sur_param,predict_sur


def get_traffic_light_param(model_config: ModelConfig) -> np.ndarray:
    N = model_config.N
    # # from vehicle CG to stopline
    # if env.engine.context.vehicle.ahead_lane_length != -1:
    #     ahead_lane_length = env.engine.context.vehicle.ahead_lane_length + \
    #                         env.engine.context.vehicle.length * 0.5
    # else:
    #     ahead_lane_length = env.engine.context.vehicle.ahead_lane_length
    # remain_phase_time = env.engine.context.vehicle.remain_phase_time
    # in_junction = env.engine.context.vehicle.in_junction
    # if ahead_lane_length < model_config.ahead_lane_length_max \
    #         and ahead_lane_length >= 0.:
    #     traffic_light = encode_traffic_light(
    #         env.engine.context.vehicle.traffic_light)
    # else:
    traffic_light = np.array([0])
    traffic_light_param = np.ones((N + 1, 3))
    traffic_light_param[:, 0] = traffic_light * np.ones((N + 1))
    traffic_light_param[:, 1] = 0 * np.ones((N + 1))
    traffic_light_param[:, 2] = 0 * np.ones((N + 1))
    return traffic_light_param


class CrossRoadContext(BaseContext):
    def advance(self, x: State) -> "CrossRoadContext":
        return CrossRoadContext(x=x, p=self.p, t=self.t, i=self.i + 1,)

    @classmethod
    def from_env(cls, env: LasvsimEnv, model_config: ModelConfig, ref_index_param: int = None):
        rng = env.np_random  # random number generator
        ego_state = env._state
        last_last_action = env.last_action  # real value
        last_action = env.action  # real value
        sumo_time = env.timestamp
        light_param = get_traffic_light_param(model_config)
        ref_param_origin = env._ref_points  # [11,4]
        ref_param = np.tile(ref_param_origin, (3, 1, 1))  # 在新的维度上复制 3 次  [3,11,4]
        # print("ref_param shape: ",ref_param.shape)
        sur_state = env._neighbor_state         #[8,7]

        sur_param =  predict_sur(sur_state,model_config)   #[21,8,7]


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
        # ego_state += ego_noise * (ego_state[4] >= 1)
        # numpy to tensor
        ego_state = torch.from_numpy(ego_state).float()  #[6]
        last_last_action = torch.from_numpy(last_last_action).float()
        last_action = torch.from_numpy(last_action).float()
        ref_param = torch.from_numpy(ref_param).float()  #[3,11,4]
        sur_param = torch.from_numpy(sur_param).float()   #[21,8,7]
        light_param = torch.from_numpy(light_param).float()  #[1]
        ref_index_param = torch.from_numpy(ref_index_param).long()

        # print("shape1: ",ego_state.shape)
        # print("shape2: ",ref_param.shape)
        # print("shape3: ",sur_param.shape)
        # print("shape4: ",light_param.shape)
        # print("shape5: ",ref_index_param.shape)

        return CrossRoadContext(
            x=State(ego_state=ego_state,
                    last_last_action=last_last_action, last_action=last_action),
            p=Parameter(ref_param=ref_param, sur_param=sur_param,
                        light_param=light_param, ref_index_param=ref_index_param,
                        ),
            t=sumo_time,
            i=0,
        )
