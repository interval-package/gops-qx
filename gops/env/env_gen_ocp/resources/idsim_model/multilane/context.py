import numpy as np
import torch

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext, State, Parameter
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig, ego_mean_array, ego_std_array, ego_lower_bound_array, ego_upper_bound_array

# from gops.env.env_gen_ocp.resources.lasvsim.lasvsim_env_qianxing import LasvsimEnv
from gops.env.env_gen_ocp.resources.lasvsim.lasvsim_env_qianxing import LasvsimEnv

from gops.env.env_gen_ocp.resources.idsim_model.multilane.ref import get_ref_param, update_ref_param
from gops.env.env_gen_ocp.resources.idsim_model.multilane.sur import get_sur_param, predict_sur
from gops.env.env_gen_ocp.resources.idsim_model.multilane.traffic import get_traffic_light_param

class MultiLaneContext(BaseContext):
    def advance(self, x: State) -> "MultiLaneContext":
        return MultiLaneContext(x=x, p=self.p, t=self.t, i=self.i + 1,)
    @classmethod
    def from_env(cls, env: LasvsimEnv, model_config: ModelConfig, ref_index_param: int = None) -> "MultiLaneContext":
        rng = np.random.default_rng()
        ego = env.lasvsim_context.ego
        ego_state = ego.state
        last_last_action = ego.last_action  # real
        last_action = ego.action  # real
        timestamp = env.timestamp

        light_param = get_traffic_light_param(env, model_config)
        ref_param = get_ref_param(env, model_config, light_param)
        sur_param = get_sur_param(env, model_config, rng)

        left_bound = ego.left_boundary_distance
        right_bound = ego.right_boundary_distance
        boundary_param = np.array([left_bound, right_bound], dtype=np.float32)
        # not update and always use green light
        # ref_param = update_ref_param(env, ref_param, light_param, model_config)
        if ref_index_param is None:
            ref_index_param = np.array(0)
        else:
            ref_index_param = np.array(ref_index_param)

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
            t=timestamp,
            i=0,
        )
