import random
from typing import Any, Dict, Tuple, List
from gops.env.env_gen_ocp.resources import idc_pb2
import gym
import numpy as np
from gym.utils import seeding
import time,math
import grpc
import torch
import time, os
import functools
from shapely.geometry import Point, LineString, Polygon
from gops.utils.math_utils import deal_with_phi_rad, convert_ref_to_ego_coord
# from gops.utils.map_tool.idc_maploader import MapBase
# from gops.utils.map_tool.idc_static_planner import IDCStaticPlanner
import matplotlib.pyplot as plt
from gops.utils.map_tool.utils import path_discrete_t_new
from gops.env.env_gen_ocp.resources.lib import point_project_to_line, compute_waypoint
from risenlighten.lasvsim.train_sim.api.trainsim import trainsim_pb2
from risenlighten.lasvsim.train_sim.api.trainsim import trainsim_pb2_grpc
from risenlighten.lasvsim.train_sim.api.trainsim import scenario_pb2
from risenlighten.lasvsim.train_sim.api.trainsim import scenario_pb2_grpc
from risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1 import train_task_pb2
from risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1 import train_task_pb2_grpc

from collections import deque

import os
from gops.utils.map_tool.idc_maploader import MapBase
from gops.utils.map_tool.lib.map import Map
from gops.env.env_gen_ocp.resources.idsim_model.utils.las_render import \
    RenderCfg, _render_tags, LasStateSurrogate, append_to_pickle_incremental, render_tags_debug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

class LoggingInterceptor(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        start_time = time.time()
        response = continuation(client_call_details, request)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        if elapsed_time > 10:  # 10 milliseconds
            pass
            # print(f"RPC call to {client_call_details.method} took {elapsed_time:.2f} ms")
        return response

def timeit(func):
    """Decorator to measure the execution time of a method."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # Start the timer
        value = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        # print(f"Function {func.__name__!r} took {elapsed_time:.4f} seconds to complete.")
        return value

    return wrapper_timer

class LasvsimEnv(gym.Env):
    def __init__(
        self,
        token: str,
        b_surr: bool = True,
        env_config: Dict = {},
        model_config: Dict = {},
        task_id = None,
        *,
        port: int = 8000,
        server_host = 'localhost:8290',
        render_info: dict = {},
        render_flag: bool = False,
        traj_flag: bool = False,
        **kwargs: Any,
    ):  
        self.port = port
        self.config = env_config
        self.metadata = [('authorization', 'Bearer ' + token)]
        self.step_counter = 0
        self.connections = dict()
        self.junctions = dict()
        self.lanes = dict()
        self.segments = dict()
        self.links = dict()
        assert task_id is not None, "None task id"
        
        self.sce_insecure_channel = grpc.insecure_channel(server_host)
        self.sce_channel = grpc.intercept_channel(self.sce_insecure_channel.__enter__(),LoggingInterceptor())
        self.sce_stub = train_task_pb2_grpc.TrainTaskStub(self.sce_channel)
        # Note the task id wiil be overwrite in the qianxing_config at "gops/env/env_gen_ocp/resources/idsim_model/params.py"
        res = self.sce_stub.GetSceneIdList(train_task_pb2.GetSceneIdListRequest(task_id = task_id), 
                                           metadata = self.metadata)
        self.scenario_list, self.version_list = res.scene_id_list, res.scene_version_list

        self.b_surr = b_surr
        self.timestamp = 0
        self.action = np.array([0,0])
        self.insecure_channel =  grpc.insecure_channel(server_host)
        self.channel =  grpc.intercept_channel(self.insecure_channel.__enter__(),LoggingInterceptor())
        self.stub = trainsim_pb2_grpc.SimulationStub(self.channel)
        self.scenario_stub = scenario_pb2_grpc.ScenarioStub(self.channel)
        self.scenario_cnt = 0

        self.render_flag = render_flag
        self.traj_flag = traj_flag
        
        try:
            self.startResp = self.stub.Init(
                trainsim_pb2.InitReq(
                    scenario_id=self.scenario_list[0],
                    scenario_version=self.version_list[0]
                ),
                metadata=self.metadata
            )
        except grpc.RpcError as e:
            print(f"gRPC error during initialization: {e}")
            raise
        self.vehicleControleReult = None
        self.stepResult = None
        self.resetResp = None
        self.last_action = np.array([0,0])
        self._ref_points = None
        self._state = None
        self._neighbor_state = None
        test_vehicle = self.stub.GetTestVehicleIdList(
            trainsim_pb2.GetTestVehicleIdListReq(
                simulation_id=self.startResp.simulation_id,
            ),
            metadata=self.metadata
        )
        self.action_lower_bound = np.array(env_config["action_lower_bound"])
        self.action_upper_bound = np.array(env_config["action_upper_bound"])
        self.action_center = (self.action_upper_bound + self.action_lower_bound) / 2
        self.action_half_range = (self.action_upper_bound - self.action_lower_bound) / 2
        self.real_action_upper = np.array(
            env_config["real_action_upper_bound"])
        self.real_action_lower = np.array(
            env_config["real_action_lower_bound"])
        self.ego_id = random.choice(test_vehicle.list)

        self.map_dict = dict()
        for i in range(len(self.scenario_list)):
            startResp = self.stub.Init(
                trainsim_pb2.InitReq(scenario_id=self.scenario_list[i],
                                     scenario_version=self.version_list[i]),
                                     metadata=self.metadata)
            cur_map = self.scenario_stub.GetHdMap(scenario_pb2.GetHdMapReq(simulation_id = startResp.simulation_id),metadata=self.metadata)
            self.map_dict[self.scenario_list[i]] = cur_map
        
        self.covert_map(self.scenario_list[0])

        vehicles_baseInfo = self.stub.GetVehicleBaseInfo(
            trainsim_pb2.GetVehicleBaseInfoReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=[self.ego_id]
            ), 
            metadata=self.metadata
        )
        self.pre_horizon = model_config["N"]
        if (self.b_surr == False):
            self.surr_veh_num = 0
        else:
            if env_config :
                self.surr_veh_num = env_config['obs_num_surrounding_vehicles']['passenger']+\
                                    env_config['obs_num_surrounding_vehicles']['bicycle'] +\
                                    env_config['obs_num_surrounding_vehicles']['pedestrian']
            else:
                self.surr_veh_num = 5

        self.ego_dim = 6
        self.ref_dim = 4 * (2*self.pre_horizon + 1)
        self.surr_dim = 8 * self.surr_veh_num
        

        # self.obs_dim = self.ego_dim + self.ref_dim + self.surr_dim - 3
        self.obs_dim = self.ego_dim + self.ref_dim - 3
        
        self.state_dim = self.ego_dim
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (self.obs_dim)),
            high=np.array([np.inf] * (self.obs_dim)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]), 
            dtype=np.float32,
        )

        self.veh_length = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.length
        self.veh_width = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.width

        self.info_dict = {
            "state": {"shape": (self.state_dim, ), "dtype": np.float32},
            "constraint": {"shape": (1,), "dtype": np.float32},
        }
        self.seed()
        self.scenario_id = self.scenario_list[0]
        self.ref_index = 0

        self._render_init(render_info=render_info)

        # Print Basic info
        # print(f"task id: {task_id}\n", token)
        # print("scenario", self.scenario_list)

  
    def step(self, action: np.ndarray,flag):
        self.step_counter += 1
        action = self.inverse_normalize_action(action)
        self.action = self.last_action + action

        self.action = np.clip(self.action, self.real_action_lower, self.real_action_upper)

        # # local dynamic rollout
        # import torch
        # from gops.env.env_gen_ocp.resources.idsim_model.model import ego_predict_model
        # if self.step_counter == 1:
        #     self._debug_dyn_state =  ego_predict_model(torch.from_numpy(self._state), torch.tensor([self.action[0], self.action[1]]), 0.1, (1800, 3058, 1.5756, 1.5756, -206369, -206369, 300, 0))
        # else: 
        #     self._debug_dyn_state =  ego_predict_model(self._debug_dyn_state, torch.tensor([self.action[0], self.action[1]]), 0.1, (1800, 3058, 1.5, 1.5, -206369, -206369, 300, 0))
            
        self.vehicleControleReult = self.stub.SetVehicleControlInfo(
            trainsim_pb2.SetVehicleControlInfoReq(
                simulation_id=self.startResp.simulation_id, 
                vehicle_id=self.ego_id, 
                lon_acc=self.action[0],
                ste_wheel=self.action[1]
            ),
            metadata=self.metadata
        )

        self.stepResult = self.stub.Step(
            trainsim_pb2.StepReq(
                simulation_id=self.startResp.simulation_id
            ), 
            metadata=self.metadata
        )
        # dynamic_info = self.stub.GetVehicleDynamicParamsInfo(
        #     trainsim_pb2.GetVehicleDynamicParamInfoReq(
        #         simulation_id=self.startResp.simulation_id,
        #         vehicle_id = self.ego_id,
        #     ),
        #     metadata=self.metadata
        # )
        # print("dynamic : ",dynamic_info)

        self.update_state()
        self.update_ref_points(flag)
        self.update_neighbor_state()

        reward, rew_info = self.reward_function_multilane()
        info = {**rew_info,**self.info}
        self.last_action  = self.action

        # obs = self.get_obs()
        obs = np.zeros_like(self.observation_space.low)
        return obs, reward, self.judge_done(), self.judge_done(),  info

    def stop(self):
        self.resetResp = self.stub.Stop(
                trainsim_pb2.StopReq(simulation_id=self.startResp.simulation_id),
                metadata=self.metadata,
            )
        
    def reset(self, expect_direction, options: dict = None, **kwargs) -> np.ndarray:
        self.step_counter = 0
        assert expect_direction in ["left", "right", "straight", "uturn"]
        test_vehicle_list = []
        if self.scenario_cnt <10:
            while len(test_vehicle_list) ==0:
                self.resetResp = self.stub.Reset(
                    trainsim_pb2.ResetReq(simulation_id=self.startResp.simulation_id,
                                          scenario_id=self.scenario_id),
                    metadata=self.metadata,
                )
                self.stepResult = self.stub.Step(
                    trainsim_pb2.StepReq(
                        simulation_id=self.startResp.simulation_id
                    ),
                    metadata=self.metadata
                )
                test_vehicle = self.stub.GetTestVehicleIdList(
                    trainsim_pb2.GetTestVehicleIdListReq(
                        simulation_id=self.startResp.simulation_id,
                    ),
                    metadata=self.metadata
                )
                if test_vehicle is not None:
                    test_vehicle_list = test_vehicle.list
            self.scenario_cnt += 1
            self.last_action = np.array([0, 0])

        else:
            while len(test_vehicle_list) ==0:
                self.resetResp = self.stub.Stop(
                    trainsim_pb2.StopReq(simulation_id=self.startResp.simulation_id),
                    metadata=self.metadata,
                )
                idx = random.randint(0, len(self.scenario_list) - 1)
                self.scenario_id = self.scenario_list[idx]
                self.startResp = self.stub.Init(
                    trainsim_pb2.InitReq(
                        scenario_id=self.scenario_id,
                        scenario_version=self.version_list[idx],
                    ),
                    metadata=self.metadata
                )
                self.stepResult = self.stub.Step(
                    trainsim_pb2.StepReq(
                        simulation_id=self.startResp.simulation_id
                    ),
                    metadata=self.metadata
                )

                test_vehicle = self.stub.GetTestVehicleIdList(
                    trainsim_pb2.GetTestVehicleIdListReq(
                        simulation_id=self.startResp.simulation_id,
                    ),
                    metadata=self.metadata
                )
                if (test_vehicle is  not None) :
                    test_vehicle_list = test_vehicle.list
                    # print("sce id2: ",self.scenario_id)
                # print("sce id2: ",test_vehicle.list)
                self.init_env(self.scenario_list[idx])
            self.scenario_cnt = 0
        
        self.ego_id = test_vehicle_list[0]
        self.update_state()
        self.update_ref_points(True)
        self.update_neighbor_state()

        return self.get_obs(), self.info

    def init_env(self,scenario_id):
        self.covert_map(scenario_id)
        self.last_action = np.array([0, 0])

        vehicles_baseInfo = self.stub.GetVehicleBaseInfo(
            trainsim_pb2.GetVehicleBaseInfoReq(
                simulation_id=self.startResp.simulation_id,
                id_list=[self.ego_id]
            ),
            metadata=self.metadata
        )

        self.veh_length = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.length
        self.veh_width = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.width

    def compute_phi(self, xt: float, xdt:float, yt: float, ydt:float) -> float:
        dx = xdt - xt
        dy = ydt - yt
        return np.arctan2(dy, dx)

    def inverse_normalize_action(self, action: np.array) -> np.array:
        action = action * self.action_half_range + self.action_center
        return action
    
    def get_obs(self) -> np.ndarray:        
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

        ego_info = self._state[None, :]
        neighbor_info = self._neighbor_state
        ref_info = self._ref_points
        
        ego_obs = np.concatenate((ego_info[:, 2], ego_info[:, 3], ego_info[:, 5]))

        ref_obs = np.concatenate(
            (
                coordinate_transformation(
                    ego_info[:, 0], ego_info[:, 1], ego_info[:, 4], ref_info[:, 0], ref_info[:, 1], ref_info[:, 2]
                ), 
                ref_info[:, 3].reshape(-1, 1)
            ), axis=1
        ).reshape(-1)  #x,y,phi,u
        
        state_egotrans = coordinate_transformation(
                    ego_info[:, 0], ego_info[:, 1], ego_info[:, 4], neighbor_info[:, 0], neighbor_info[:, 1], neighbor_info[:, 2]
                )
        neighbor_obs = np.concatenate(
            (state_egotrans[:, :2], np.cos(state_egotrans[:, 2:3]), np.sin(state_egotrans[:, 2:3]), neighbor_info[:, 3:]), axis=1
        ).reshape(-1)

        obs = np.concatenate((ego_obs, ref_obs, neighbor_obs))
        obs = obs.astype(np.float32)
        return obs


    ### Render utils

    #### Core params
    _render_count = 0
    _render_tags = _render_tags
    _render_tags_debug = render_tags_debug
    _render_cfg: RenderCfg
    render_flag: bool
    traj_flag: bool

    ####  data buffers    
    _render_info = {}
    _render_ego_shadows = deque([])
    _render_surcars:list
    _render_done_info = {}

    def _render_init(self, render_info):
        if not self.render_flag and not self.traj_flag:
            print("Without pic and data saved")
            return
        else:
            print(f"Into the data verbose mode. render: {self.render_flag}, traj: {self.traj_flag}")

        _map = Map()
        # New interface not support hdmap obj
        _map.load_hd(self.map_dict[self.scenario_list[0]])

        path_flag = render_info.get("_debug_path_qxdata", None)
        if path_flag is None:
            policy = render_info.get("policy", None)
            if policy is not None:
                _debug_path_qxdata = f"./data_qx/{'draw' if self.render_flag else 'data'}/{policy}/" \
                    + time.strftime("%m-%d-%H:%M:%S")
            else:
                # In the training mode
                _debug_path_qxdata = f"./data_qx/train/" \
                    + time.strftime("%m-%d-%H:%M:%S")
            
            # The first time is init, the inited info is get by the path key
            # Warning should assert that it's shared
            render_info["_debug_path_qxdata"] = _debug_path_qxdata
            os.makedirs(_debug_path_qxdata, exist_ok=True)

            self._render_cfg = RenderCfg()
            self._render_cfg.set_vals(**{
                "_debug_path_qxdata": _debug_path_qxdata,
                "show_npc": render_info["show_npc"],
                "draw_bound": render_info["draw_bound"],
                # "map": _map,
                "render_type": render_info["type"], # pic type
                "render_config": render_info,
            })
            self._render_cfg.save()
        else:
            _debug_path_qxdata = path_flag
            self._render_cfg = RenderCfg()
            self._render_cfg.set_vals(**{
                "_debug_path_qxdata": _debug_path_qxdata,
                "show_npc": render_info["show_npc"],
                "draw_bound": render_info["draw_bound"],
                "map": _map,
                "render_type": render_info["type"], # pic type
                "render_config": render_info,
            })
            # self._render_cfg.save()

        if self.render_flag:
            f = plt.figure(figsize=(16,9))
            # draw totoal map
            _map.draw()
            plt.cla()
            
            f.subplots_adjust(left=0.25)
            _map.draw_everything(show_id=False, show_link_boundary=False)

    def _render_parse_surcar(self, around_moving_objs):
        ret = []

        for neighbor in around_moving_objs:
            row =\
            [
                neighbor.position.point.x,
                neighbor.position.point.y,
                neighbor.base_info.length,
                neighbor.base_info.width,
                neighbor.position.phi,
                getattr(neighbor.base_info, "obj_id", "none")
            ]
            ret.append(row)

        self._render_surcars = ret

    def _render_save_traj(self):

        _debug_adaptive_vars = {
            # "_debug_dyn_state"    : self._debug_dyn_state.numpy(),
            "_debug_done_errlat"  : self._debug_done_errlat ,
            # "_debug_done_errlon"  : self._debug_done_errlon ,
            "_debug_done_errhead" : self._debug_done_errhead,
            "_debug_done_postype" : self._debug_done_postype,
            "_debug_reward_scaled_punish_boundary": self._debug_reward_scaled_punish_boundary,
        }

        # filter save, make a triger to save vital cases.
        obj = \
        LasStateSurrogate(
            # Basic draw
            self._state,
            self._ref_points,
            self.action,
            self._ego[:6].astype(np.float32),
            self._render_surcars,
            self._render_info,
            self._render_done_info,
            _debug_adaptive_vars,

            # Dynamic test
            # self._debug_dyn_state.numpy()
        )
        
        append_to_pickle_incremental(os.path.join(self._render_cfg._debug_path_qxdata, "trajs.pkl"), obj)
        pass

    def _render_update_info(self, mf_info, *, add_info={}):
        self._render_info = {}
        for tag in self._render_tags:
            if tag in mf_info.keys() and mf_info[tag] is not None:
                self._render_info[tag] = mf_info[tag]
            # self._render_info.update(add_info)
    
    def _render_sur_byobs(self, neighbor_info=None, color = 'black', *, save_func=None, show_done=True, show_debug=False, **kwargs):
        if self.traj_flag:
            self._render_save_traj()
        if not self.render_flag:
            return
        
        original_nei = self._render_surcars
        
        f, ax = plt.gcf(), plt.gca()
        ego_x, ego_y = self._state[0], self._state[1]
        phi = self._state[4]
        dx, dy = self._render_cfg.arrow_len*np.cos(phi), self._render_cfg.arrow_len*np.sin(phi)
        arrow = plt.arrow(ego_x, ego_y, dx, dy, head_width=0.5)
        plt.xlim(ego_x-self._render_cfg.draw_bound, ego_x+self._render_cfg.draw_bound)
        plt.ylim(ego_y-self._render_cfg.draw_bound, ego_y+self._render_cfg.draw_bound)
        dot = plt.scatter(ego_x, ego_y, color='red', s=10)
        self._render_ego_shadows.append((dot, arrow))
        
        ref_x, ref_y = self._ref_points[:, 0], self._ref_points[:, 1]
        ref_lines = plt.plot(ref_x, ref_y, ls="dotted", color="red", linewidth=8)
        
        # reward, reward_info = self._buffered_reward
        text_strs = [
            f"Ego_speed: {self._state[2]:.2f}", 
            f"act: {self.action[0]:.2f}, {self.action[1] * 180 / np.pi:.2f}",
            f"yaw rate: {self._state[5]:.3f}",
            f"vx, vy, phi: {self._state[2]:.2f}, {self._state[3]:.2f}, {self._state[4]:.2f}",
            f"pos: {self._state[0]:.2f}, {self._state[1]:.2f}",
            # f"reward: {reward}"
            ] + [f"{key}: {val:.2f}" for key, val in self._render_info.items()]
        
        if show_done:
            text_strs += [f"{key}:{val};" for key, val in self._render_done_info.items()]
        if show_debug:
            text_strs += [f"{key}:{self[key]};" for key in self._render_tags_debug]

        height = 1/len(text_strs)
        text_locs = [(0.2, 1-height*i) for i, _ in enumerate(text_strs)]
        text_objs = []
        for text_str, text_loc in zip(text_strs, text_locs):
            text_obj = f.text(
                text_loc[0], text_loc[1],  
                text_str,
                fontsize=10,  
                fontweight='bold',  
                ha='right',  # 
                va='top',  # 
                color= color  # 
            )
            text_objs.append(text_obj)
        
        def draw_car(center_x, center_y, length, width, phi, facecolor="lightblue", id=None):
            car = patches.Rectangle(
                (center_x - length / 2, center_y - width / 2),  # Bottom-left corner of the rectangle
                length,  # Width
                width,   # Height
                angle=np.degrees(phi),  # Rotation angle in degrees
                edgecolor='black',
                rotation_point="center",
                facecolor=facecolor,
                alpha=0.5
            )
            text = None
            if self._render_cfg.show_npc:
                info = f"({center_x:.2f}, {center_y:.2f}): {phi:.2f}"
                if id is not None:
                    info = f"{id}: " + info
                text = ax.text(center_x, center_y, info, fontsize=13)
            return car, text

        def remove_car(car_t):
            car, text = car_t
            car.remove()
            if text is not None:
                text.remove()
        
        car_rectangles = []

        for index, neighbor in enumerate(original_nei):
            
            center_x, center_y, length, width, phi, obj_id = neighbor
            
            if ego_x-self._render_cfg.draw_bound <= center_x and ego_x+self._render_cfg.draw_bound >= center_x and ego_y-self._render_cfg.draw_bound <= center_y and ego_y+self._render_cfg.draw_bound >= center_y:
                car_rectangles.append(draw_car(center_x, center_y, length, width, phi, obj_id))
                plt.gca().add_patch(car_rectangles[-1][0])
            
        ego_car_t = draw_car(float(self._ego[0]), float(self._ego[1]), float(self._ego[4]), float(self._ego[5]), self._state[4], "pink", "ego")
        plt.gca().add_patch(ego_car_t[0])
        car_rectangles.append(ego_car_t)
        
        # saving
        if save_func is None:
            self._render_count += 1 
            f.savefig(os.path.join(f"{self._render_cfg._debug_path_qxdata}", str(self._render_count) + self._render_cfg.render_type), dpi=self._render_cfg["dpi"])
        else:
            save_func(f, ax)

        # Cleaning
        for text_obj in text_objs:
            text_obj.remove()
        for car in car_rectangles:
            remove_car(car)
        for line in ref_lines:
            line.remove()
            
        if len(self._render_ego_shadows) > 30:
            ego = self._render_ego_shadows.popleft()
            for i in ego:
                i.remove()
        return

    def model_free_reward(self,
                            context, # S_t
                            last_last_action, # absolute action, A_{t-2}
                            last_action, # absolute action, A_{t-1}
                            action # normalized incremental action, _t - A_{t-1} / Z
                            ) -> Tuple[float, dict]:
        # all inputs are batched
        # vehicle state: context.x.ego_state
        ego_state = context.x.ego_state[0] # [6]: x, y, phi, u, v, w
        ref_param = context.p.ref_param[0] # [R, 2N+1, 4] ref_x, ref_y, ref_phi, ref_v
        ref_index = context.p.ref_index_param[0]
        ref_state = ref_param[ref_index, context.i, :] # 4
        next_ref_state = ref_param[ref_index, context.i + 1, :] # 4
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state
        ref_x, ref_y, ref_phi, ref_v = ref_state
        next_ref_v = next_ref_state[3]
        last_acc, last_steer = last_action[0][0], last_action[0][1]*180/np.pi
        last_last_acc, last_last_steer = last_last_action[0][0], last_last_action[0][1]*180/np.pi
        delta_steer =  (last_steer - last_last_steer)/self.config['dt']
        jerk = (last_acc - last_last_acc)/self.config['dt']
        # print(f'last_acc: {last_acc}, last_steer: {last_steer}, delta_steer: {delta_steer}, jerk: {jerk}')

        # live reward
        rew_step =  1.0

        tracking_error = np.sqrt((ego_x - ref_x) ** 2 + (ego_y - ref_y) ** 2) 
        delta_phi = deal_with_phi_rad(ego_phi - ref_phi) * 180 / np.pi # degree
        ego_r = ego_r * 180/np.pi # degree
        speed_error = ego_vx - ref_v
        self.out_of_range = self.out_of_range or (abs(speed_error) > 1)
        # tracking_error
        punish_dist_lat = 5*np.where(
            np.abs(tracking_error) < 0.3,
            np.square(tracking_error),
           0.02* np.abs(tracking_error) + 0.084,
        ) # 0~1 0~6m 50% 0~0.3m

        punish_vel_long = 0.5 * np.where(
            np.abs(speed_error) < 1,
            np.square(speed_error),
            0.1*np.abs(speed_error)+0.85,
        ) # 0~1 0~11.5m/s 50% 0~1m/s
        punish_head_ang = 0.05*np.where(
            np.abs(delta_phi) < 3,
            np.square(delta_phi),
            np.abs(delta_phi)+ 8,
        ) # 0~1  0~12 degree 50% 0~3 degree

        punish_yaw_rate = 0.1*np.where(
            np.abs(ego_r) < 2,
            np.square(ego_r),
            np.abs(ego_r)+ 2,
        ) # 0~1  0~8 degree/s 50% 0~2 degree/s

        scaled_punish_overspeed = 3*np.clip(
        np.where(
            ego_vx > 1.1*ref_v,
            1 + np.abs(ego_vx - 1.1*ref_v),
            0,),
        0, 2)

        scaled_punish_dist_lat = punish_dist_lat * self.config['P_lat']
        scaled_punish_vel_long = punish_vel_long * self.config['P_long']
        scaled_punish_head_ang = punish_head_ang * self.config['P_phi']
        scaled_punish_yaw_rate = punish_yaw_rate * self.config['P_yaw']

        # reward related to action
        # nominal_steer = self._get_nominal_steer_by_state(
        #     ego_state, ref_param, ref_index)*180/np.pi * 0.0
        # print(f'nominal_steer: {nominal_steer}')
        nominal_steer = 0.0

        abs_steer = np.abs(last_steer- nominal_steer)   
        reward_steering = -np.where(abs_steer < 4, np.square(abs_steer), 2*abs_steer+8)

        self.out_of_action_range = abs_steer > 20

        if ego_vx < 1 and self.config['enable_slow_reward']:
            reward_steering = reward_steering * 5

        abs_ax = np.abs(last_acc)
        reward_acc_long = -np.where(abs_ax < 2, np.square(abs_ax), 2*abs_ax) * 0.0

        reward_delta_steer = -np.where(np.abs(delta_steer) < 4, np.square(delta_steer), 2*np.abs(delta_steer)+8)
        reward_jerk = -np.where(np.abs(jerk) < 2, np.square(jerk), 2*np.abs(jerk)+8)

        self.turning = np.abs(nominal_steer) > 5 and self.in_junction
        if self.turning:
            scaled_punish_dist_lat = scaled_punish_dist_lat * 0.5
            scaled_punish_head_ang = scaled_punish_head_ang 
            scaled_punish_yaw_rate = scaled_punish_yaw_rate * 0.2
            scaled_punish_vel_long = scaled_punish_vel_long * 0.2
            reward_steering = reward_steering * 0.2
            rew_step = np.clip(ego_vx, 0, 1.0)*2

        # if ego_vx < 1 and self.config.enable_slow_reward:
        #     scaled_punish_dist_lat = scaled_punish_dist_lat * 0.1
        #     scaled_punish_head_ang = scaled_punish_head_ang * 0.1
        break_condition = (ref_v < 2 and (next_ref_v - ref_v) < -0.1) or (ref_v < 1.0)
        if break_condition and self.config['nonimal_acc']:
            nominal_acc = -2.5
            # scaled_punish_vel_long = 0  # remove the effect of speed error
            scaled_punish_dist_lat = 0  # remove the effect of tracking error
            scaled_punish_head_ang = 0
            reward_acc_long = 0
        else:
            nominal_acc = 0
            punish_nominal_acc = 0

        delta_acc = np.abs(nominal_acc - last_acc)
        punish_nominal_acc =(nominal_acc != 0)* 4 * np.where(delta_acc < 0.5, np.square(delta_acc),  delta_acc-0.25)

        # action related reward
        scaled_reward_steering = reward_steering * self.config['P_steer']
        scaled_reward_acc_long = reward_acc_long * self.config['P_acc']
        scaled_reward_delta_steer = reward_delta_steer * self.config['P_delta_steer']

        scaled_reward_jerk = reward_jerk * self.config['P_jerk']

        # live reward
        scaled_rew_step = rew_step * self.config['R_step']

        reward_ego_state = scaled_rew_step - \
            (scaled_punish_dist_lat + 
             scaled_punish_vel_long + 
             scaled_punish_head_ang + 
             scaled_punish_yaw_rate + 
             punish_nominal_acc + 
             scaled_punish_overspeed) + \
            (scaled_reward_steering + 
             scaled_reward_acc_long + 
             scaled_reward_delta_steer + 
             scaled_reward_jerk)
        # print("reward ego state1: ",reward_ego_state)
        reward_ego_state = np.clip(reward_ego_state, -50, 20)
        # print("reward ego state2: ",reward_ego_state)

        return reward_ego_state, {
            "env_tracking_error": tracking_error,
            "env_speed_error": np.abs(speed_error),
            "env_delta_phi": np.abs(delta_phi),

            "env_reward_step": rew_step,

            "env_reward_steering": reward_steering,
            "env_reward_acc_long": reward_acc_long,
            "env_reward_delta_steer": reward_delta_steer,
            "env_reward_jerk": reward_jerk,

            "env_reward_dist_lat": -punish_dist_lat,
            "env_reward_vel_long": -punish_vel_long,
            "env_reward_head_ang": -punish_head_ang,
            "env_reward_yaw_rate": -punish_yaw_rate,

            "env_scaled_reward_part2": reward_ego_state,
            "env_scaled_reward_step": scaled_rew_step,
            "env_scaled_reward_dist_lat": -scaled_punish_dist_lat,
            "env_scaled_reward_vel_long": -scaled_punish_vel_long,
            "env_scaled_reward_head_ang": -scaled_punish_head_ang,
            "env_scaled_reward_yaw_rate": -scaled_punish_yaw_rate,
            "env_scaled_reward_steering": scaled_reward_steering,
            "env_scaled_reward_acc_long": scaled_reward_acc_long,
            "env_scaled_reward_delta_steer": scaled_reward_delta_steer,
            "env_scaled_reward_jerk": scaled_reward_jerk,
        }

    def reward_function_multilane(self):
        def coordinate_transformation(x_0, y_0, phi_0, x, y, phi):
            x_0 = float(x_0)
            y_0 = float(y_0)
            phi_0 = float(phi_0)
            x_ = (x - float(x_0)) * np.cos(float(phi_0)) + (y- float(y_0)) * np.sin(float(phi_0))
            y_ = -(x - x_0) * np.sin(phi_0) + (y - y_0) * np.cos(phi_0)
            phi_ = phi - phi_0
            return np.stack((x_, y_, phi_), axis=-1)

        ego_x, ego_y = self._state[:2]#vehicle.ground_position
        # tracking_error cost
        ref = LineString(self._ref_points[:,:2])
        position_on_ref = point_project_to_line(ref, ego_x, ego_y)
        current_first_ref_x, current_first_ref_y, current_first_ref_phi = compute_waypoint(ref, position_on_ref)

        tracking_error = np.sqrt((ego_x - current_first_ref_x) **2 + (ego_y - current_first_ref_y) **2 ) 

        delta_phi = deal_with_phi_rad(self._state[4] - current_first_ref_phi)

        pos = self.stub.GetVehiclePosition(trainsim_pb2.GetVehiclePositionReq(simulation_id=self.startResp.simulation_id,
                id_list=[self.ego_id]), metadata=self.metadata)
        ego_pos = pos.position_dict.get(self.ego_id).position_type
        if ego_pos == 2:
            self.in_junction = True
        else:
            self.in_junction = False
            
        tracking_error_lat = -math.sin(current_first_ref_phi) * (ego_x - current_first_ref_x) + math.cos(current_first_ref_phi) * (ego_y - current_first_ref_y)
        # tracking_error_long = math.cos(current_first_ref_phi) * (ego_x - current_first_ref_x) + math.sin(current_first_ref_phi) * (ego_y - current_first_ref_y)
        self.out_of_range = (tracking_error_lat > 8) \
                            or (np.abs(delta_phi) > np.pi/2) \
                            # or (tracking_error_long > 2) \
                            # or (ego_pos == 1)
        self._debug_done_errlat  = tracking_error_lat > 8
        # self._debug_done_errlon  = tracking_error_long > 2
        self._debug_done_errhead = np.abs(delta_phi) > np.pi/4
        self._debug_done_postype = ego_pos == 1
        
        # self.out_of_range = tracking_error > 8 or np.abs(delta_phi) > np.pi/4 or ego_pos == 1

        # collision risk cost
        ego_vx = float(self._ego[3])
        ego_W = float(self._ego[5])
        ego_L = float(self._ego[4])
        head_ang_error = delta_phi

        safety_lat_margin_front = self.config['safety_lat_margin_front']
        safety_lat_margin_rear = safety_lat_margin_front # TODO: safety_lat_margin_rear
        safety_long_margin_front = self.config['safety_long_margin_front']
        safety_long_margin_side = self.config['safety_long_margin_side']
        front_dist_thd = self.config['front_dist_thd']
        space_dist_thd = self.config['space_dist_thd']
        rel_v_thd = self.config['rel_v_thd']
        rel_v_rear_thd = self.config['rel_v_rear_thd']
        time_dist = self.config['time_dist']

        punish_done = self.config['P_done']

        pun2front = 0.
        pun2side = 0.
        pun2space = 0.
        pun2rear = 0.

        pun2front_sum = 0.
        pun2side_sum = 0.
        pun2space_sum = 0.
        pun2rear_sum = 0.

        min_front_dist = np.inf

        sur_info = self._neighbor_vehicle
        ego_edge = self._ego[6]
        ego_lane = self._ego[7]

        for sur_vehicle in sur_info:
            mask = sur_vehicle[-2]
            if mask == 0:
                continue
            sur_x = float(sur_vehicle[0])
            sur_y = float(sur_vehicle[1])
            sur_phi = float(sur_vehicle[2])
            rel_x,rel_y,rel_phi = coordinate_transformation(self._state[0],self._state[1],self._state[4],sur_x,sur_y,sur_phi)
            # print(rel_x,rel_y,rel_phi)
            sur_vx = float(sur_vehicle[3])
            sur_lane = sur_vehicle[-1]

            sur_W = float(sur_vehicle[4])
            sur_L = float(sur_vehicle[5])
            # [1 - tanh(x)]: 0.25-> 75%  0.5->54%, 1->24%, 1.5->9.5% 2->3.6%, 3->0.5%
            if np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front and rel_x > 0:
                min_front_dist = min(min_front_dist, rel_x - (ego_L + sur_L) / 2)  

            pun2front_cur = np.where( (sur_lane == ego_lane or  np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front)
                 and rel_x >= 0 and rel_x < front_dist_thd and ego_vx > sur_vx,
                np.clip(1. - np.tanh((rel_x-(ego_L + sur_L) / 2 - safety_long_margin_front) / (time_dist*(np.max(ego_vx,0) + 0.1))), 0., 1.),
                0,
            )
            pun2front = np.maximum(pun2front, pun2front_cur)
            pun2front_sum += pun2front_cur

            pun2side_cur =  np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 + safety_long_margin_side and rel_y*head_ang_error > 0 and  rel_y > (ego_W + sur_W) / 2, 
                np.clip(1. - np.tanh((np.abs(rel_y)- (ego_W + sur_W) / 2) / (np.abs(ego_vx*np.sin(head_ang_error))+0.01)), 0., 1.),
                0,
            )
            pun2side = np.maximum(pun2side, pun2side_cur)
            pun2side_sum += pun2side_cur

            pun2space_cur = np.where(
                np.abs(rel_y) < (ego_W + sur_W) / 2 and rel_x >= 0 and rel_x < space_dist_thd and ego_vx > sur_vx + rel_v_thd,
                np.clip(1. - (rel_x - (ego_L + sur_L) / 2) / (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,) + np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 and np.abs(rel_y) > (ego_W + sur_W) / 2,
                np.clip(1. - np.tanh(3.0*(np.abs(rel_y) - (ego_W + sur_W) / 2)), 0., 1.),
                0,)
            pun2space = np.maximum(pun2space, pun2space_cur)
            pun2space_sum += pun2space_cur

            pun2rear_cur = np.where(
                (sur_lane == ego_lane or  np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_rear) and rel_x < 0 and rel_x > -space_dist_thd and ego_vx < sur_vx - rel_v_rear_thd,
                np.clip(1. - (-1)*(rel_x + (ego_L + sur_L) / 2) / (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,)
            pun2rear = np.maximum(pun2rear, pun2rear_cur)
            pun2rear_sum += pun2rear_cur

        '''if self.config.punish_sur_mode == "sum":
            pun2front = pun2front_sum
            pun2side = pun2side_sum
            pun2space = pun2space_sum
            pun2rear = pun2rear_sum
        elif self.config.punish_sur_mode == "max":
            pass
        else:
            raise ValueError(f"Invalid punish_sur_mode: {self.config.punish_sur_mode}")'''
        #print("config:",self.config)
        scaled_pun2front = pun2front * self.config['P_front']
        scaled_pun2side = pun2side * self.config['P_side']
        scaled_pun2space = pun2space * self.config['P_space']
        scaled_pun2rear = pun2rear * self.config['P_rear']
        self.braking_mode = (min_front_dist < 4)

        punish_collision_risk = scaled_pun2front + scaled_pun2side + scaled_pun2space + scaled_pun2rear

        # out of driving area cost
        if self.in_junction or self.config['P_boundary'] == 0: # TODO: boundary cost = 0  when boundary info is not available
            punish_boundary = 0.
        else:
            rel_angle = np.abs(delta_phi)
            left_distance,right_distance = self.get_boundary_distance()
            if left_distance+right_distance<4:
                min_left_distance = left_distance - (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
                min_right_distance = right_distance - (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
                boundary_safe_margin = 0.5
                boundary_distance = np.clip(np.minimum(min_left_distance, min_right_distance), 0.,None)

                punish_boundary = np.where(
                    boundary_distance < boundary_safe_margin,
                    np.clip((1. - boundary_distance/boundary_safe_margin), 0., 1.),
                    0.0,
                )
            else:
                self.out_of_range = True
                punish_boundary = 1

        scaled_punish_boundary = punish_boundary * self.config['P_boundary']
        self._debug_reward_scaled_punish_boundary = scaled_punish_boundary
        # action related reward

        reward = - scaled_punish_boundary

        if self.config['penalize_collision']:
            reward -= punish_collision_risk

        event_flag = 0 # nomal driving (on lane, stop)
        # Event reward: target reached, collision, out of driving area
        if ego_vx < 1 and  not self.braking_mode:  # start to move from stop
            event_flag = 1
        if self.braking_mode:  # start to brake
            event_flag = 2

        collision_info = self.stub.GetVehicleCollisionInfo(
            trainsim_pb2.GetVehicleCollisionInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id= self.ego_id,
            ),
            metadata=self.metadata
        )

        collision_flag = collision_info.collision_flag
        # if collision_flag:   # collision
        #     reward -= punish_done if self.config['penalize_collision'] else 0.
        #     event_flag = 3
        if self.out_of_range or collision_flag:  # out of driving area
            reward -= punish_done
            event_flag = 4

        return reward, {
            "category": event_flag,
            "env_pun2front": pun2front,
            "env_pun2side": pun2side,
            "env_pun2space": pun2space,
            "env_pun2rear": pun2rear,
            "env_scaled_reward_part1": reward,
            "env_reward_collision_risk": -punish_collision_risk, #- punish_collision_risk,
            "env_scaled_pun2front": scaled_pun2front, #scaled_pun2front,
            "env_scaled_pun2side": scaled_pun2side, #scaled_pun2side,
            "env_scaled_pun2space": scaled_pun2space, #scaled_pun2space,
            "env_scaled_pun2rear": scaled_pun2rear, #scaled_pun2rear,
            "env_scaled_punish_boundary": scaled_punish_boundary, #scaled_punish_boundary,
        }

    def get_constraint(self) -> np.ndarray:
        # TODO: implement this, get constraint from self.stub
        return np.random.uniform(low=-1, high=1, size=(1,))

    def judge_done(self) -> bool:
    
        collision_info = self.stub.GetVehicleCollisionInfo(
            trainsim_pb2.GetVehicleCollisionInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id=self.ego_id,
            ),
            metadata=self.metadata
        )
        collision_flag = collision_info.collision_flag
        if collision_flag:
            return True

        pos = self.stub.GetVehiclePosition(
            trainsim_pb2.GetVehiclePositionReq(simulation_id=self.startResp.simulation_id,
                                               id_list=[self.ego_id]), metadata=self.metadata)
        ego_pos = pos.position_dict.get(self.ego_id).position_type

        done =  (ego_pos == 1)
        park_flag = self._ego[3] == 0.
        tracking_out_of_region = self.out_of_range
        self._render_done_info = {
            "Pause": park_flag,
            "RegionOut": tracking_out_of_region,
            "Collision": collision_flag,
            "MapOut": done
        }
        if tracking_out_of_region or park_flag:
            done = True
        return done
    
    def update_state(self):
        '''
        return: [x, y, phi, u, v, w]
        '''

        vehicles_id_list = [self.ego_id]

        vehicles_position = self.stub.GetVehiclePosition(
            trainsim_pb2.GetVehiclePositionReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )

        vehicles_baseInfo = self.stub.GetVehicleBaseInfo(
            trainsim_pb2.GetVehicleBaseInfoReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )

        vehicles_MovingInfo = self.stub.GetVehicleMovingInfo(
            trainsim_pb2.GetVehicleMovingInfoReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )

        x = vehicles_position.position_dict.get(self.ego_id).point.x
        y = vehicles_position.position_dict.get(self.ego_id).point.y
        phi = vehicles_position.position_dict.get(self.ego_id).phi
        u = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).u
        v = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).v
        w = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).w
        self._state = np.array([x, y, u, v, phi, w])
        length = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.length
        width = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.width
        junction_id = vehicles_position.position_dict.get(self.ego_id).junction_id
        lane_id = vehicles_position.position_dict.get(self.ego_id).lane_id
        link_id = vehicles_position.position_dict.get(self.ego_id).link_id
        segment_id = vehicles_position.position_dict.get(self.ego_id).segment_id
        mask = 1
        self._ego = np.array([x, y, phi, u, length, width , segment_id, junction_id, lane_id, link_id ])


    def update_ref_points(self,flag):
        refe_points = self.stub.GetVehicleReferenceLines(trainsim_pb2.GetVehicleReferenceLinesReq(
            simulation_id=self.startResp.simulation_id, vehicle_id=self.ego_id), metadata=self.metadata)
        
        ref_lines = refe_points.reference_lines
        u = self._clac_ref_u(refpoints=ref_lines)
        if len(ref_lines) > 0:
            if len(ref_lines[0].points)>1:
                phi = self.compute_phi(ref_lines[0].points[0].x, ref_lines[0].points[1].x, ref_lines[0].points[0].y, ref_lines[0].points[1].y)
        else:
            phi = 0
        if len(ref_lines)>0:
            if flag:
                self.ref_index = np.clip(self.ref_index + (1 if random.random() > 0.5 else -1), 0, len(ref_lines)-1) if len(ref_lines) >= 1 else 0
            if (self.ref_index > len(ref_lines)-1) or len(ref_lines)==1:
                self.ref_index = 0
            ref_line = np.array([[point.x, point.y, phi, u] for point in ref_lines[self.ref_index].points]).T
            ref_path_discrete = path_discrete_t_new(ref_line, 0.1, self.ref_dim)
            ref_points = []
            for i in range(2*self.pre_horizon + 1):
                ref_x = ref_path_discrete[0][i]
                ref_y = ref_path_discrete[1][i]
                ref_phi = ref_path_discrete[2][i]
                ref_u = ref_path_discrete[3][i]
                ref_points.append([ref_x, ref_y, ref_phi, ref_u])
            self._ref_points = np.array(ref_points, dtype=np.float32)
            # print("ref points: ",self._ref_points[0],self._ref_points[1],self._ref_points[2])


    def _clac_ref_u(self, refpoints):
        return 10
        # TODO u should be a list, change this func reference.

    # def update_ref_points_origin(self):
    #     '''
    #     return: [ref_x, ref_y, ref_phi, ref_u] * (pre_horizon + 1)
    #     '''
    #     # TODO: Confirm the time interval of trajectory points in lasvsim.
    #     vehicle = self.stub.GetVehicle(
    #         trainsim_pb2.GetVehicleReq(
    #             simulation_id=self.startResp.simulation_id,
    #             vehicle_id=self.ego_id
    #         ),
    #         metadata=self.metadata
    #     )
    #     # TODO ref_u
    #     ref_u = 5
    #     if vehicle.vehicle.info.static_path == []:
    #         ref_x = vehicle.vehicle.info.moving_info.position.point.x
    #         ref_y = vehicle.vehicle.info.moving_info.position.point.y
    #         ref_phi = vehicle.vehicle.info.moving_info.position.phi
    #         ref_points = [[ref_x, ref_y, ref_phi, ref_u]] * (self.pre_horizon + 1)
    #     else:
    #         ref_points = []
    #         for i in range(self.pre_horizon + 1):
    #             ref_x = vehicle.vehicle.info.static_path[0].point[i].x
    #             ref_dx = vehicle.vehicle.info.static_path[0].point[i + 1].x
    #             ref_y = vehicle.vehicle.info.static_path[0].point[i].y
    #             ref_dy = vehicle.vehicle.info.static_path[0].point[i + 1].y
    #             ref_phi = self.compute_phi(ref_x, ref_dx, ref_y, ref_dy)
    #             ref_points.append([ref_x, ref_y, ref_phi, ref_u])
    #     self._ref_points = np.array(ref_points, dtype=np.float32)

    
    def update_neighbor_state(self):
        '''
        return: [x, y, phi, speed, Length, Width, mask, laneid] * self.surr_veh_num
        '''
        distances = []
        neighbor_info = []

        def CalDist(x1, y1, x2, y2):
            return ((x1-x2)**2+(y1-y2)**2)**0.5

        # get the indices of the smallest k elements
        def get_indices_of_k_smallest(arr, k):
            idx = np.argpartition(arr, k)
            return idx[:k]

        def coordinate_transformation(x_0, y_0, phi_0, x, y, phi):
            x_0 = float(x_0)
            y_0 = float(y_0)
            phi_0 = float(phi_0)
            x_ = (x - float(x_0)) * np.cos(float(phi_0)) + (y - float(y_0)) * np.sin(float(phi_0))
            y_ = -(x - x_0) * np.sin(phi_0) + (y - y_0) * np.cos(phi_0)
            phi_ = phi - phi_0
            return np.stack((x_, y_, phi_), axis=-1)

        # TODO: 鎰熺煡鑾峰彇

        perception_info = self.stub.GetVehiclePerceptionInfo(
            trainsim_pb2.GetVehiclePerceptionInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id = self.ego_id,
            ),
            metadata=self.metadata
        )
        around_moving_objs = perception_info.list

        self._render_parse_surcar(around_moving_objs)

        # print("perception res: ",around_moving_objs)

        # Preprocess the data of neighbor vehicles
        if (self.b_surr):
            if len(around_moving_objs) > 0:
                n = len(around_moving_objs)
                for i in range(n):
                    dist = CalDist(around_moving_objs[i].position.point.x,
                                   around_moving_objs[i].position.point.y,
                                   self._state[0],
                                   self._state[1])
                    distances.append(dist)

            # sort out the smallest k distance vehicles
            if (len(distances) > self.surr_veh_num):
                indices = get_indices_of_k_smallest(distances, self.surr_veh_num)
            else:
                indices = range(len(distances))

            def __add_info(val):
                val = float(val)
                neighbor_info.append(val)

            # append info of the smallest k distance vehicles
            for i in indices:
                if np.abs(self.state[0] - around_moving_objs[i].position.point.x) > 0.001 or np.abs(self.state[1] - around_moving_objs[i].position.point.y) > 0.001:
                    # Caution: The idx 0 of around is ignored due to the nearset car currently is the car itself
                    sur_x, sur_y, sur_phi = around_moving_objs[i].position.point.x, \
                        around_moving_objs[i].position.point.y, \
                        around_moving_objs[i].position.phi
                    rel_x, rel_y, rel_phi = coordinate_transformation(self._ego[0], self._ego[1], self._ego[2], 
                                                                      sur_x, sur_y, sur_phi)
                    if rel_phi < np.pi/2:
                        __add_info(around_moving_objs[i].position.point.x)
                        __add_info(around_moving_objs[i].position.point.y)
                        __add_info(around_moving_objs[i].position.phi)
                        __add_info(around_moving_objs[i].moving_info.u)
                        # __add_info(vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).v)
                        # __add_info(vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).w)
                        __add_info(around_moving_objs[i].base_info.length)
                        __add_info(around_moving_objs[i].base_info.width)
                        __add_info(1) # mask, 1 indicate the real car and the 0 indicate the virtual one
                        neighbor_info.append(around_moving_objs[i].position.lane_id)

            # append 0 if the number of neighbor vehicles is less than 5
            if (len(neighbor_info) < self.surr_dim):
                neighbor_info.extend([0]*(self.surr_dim-len(neighbor_info)))

        self._neighbor_vehicle = np.array(neighbor_info).reshape(-1, 8)

        self._neighbor_state = self._neighbor_vehicle[:,:7]
        #print("update35: ",time.time()-time35)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def ref_points(self) -> np.ndarray:
        return self._ref_points

    @property
    def neighbor_state(self) -> np.ndarray:
        return self._neighbor_state

    @property
    def info(self) -> Dict:
        return {
            "state": self._state,
            "constraint": self.get_constraint(),
            # "ref_path": self.xy_points,
        }

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_error(self, err):
        if err is None:
            return False

        if err.code != 0:
            print(err.msg)
            return True
        return False

    def __del__(self):
        self.insecure_channel.__exit__(None, None, None)

    def fill_protobuf_data(self,input_data):

        def CalDist(x1, y1, x2, y2):
            return ((x1-x2)**2+(y1-y2)**2)**0.5
        def id_to_numbers(id_string):
            numbers = []
            for char in id_string:
                if char.isdigit():  # Check if the character is a digit
                    numbers.append(char)
                elif char.isalpha():  # Check if the character is a letter
                    number = ord(char.lower()) - ord('a') + 1  # Convert letter to number
                    numbers.extend(list(str(number)))  # Extend the list with the digits of the number
                else:
                    continue  # Skip other characters
            return np.int64(''.join(numbers))

        def obstacle_type_to_int(obstacle_type_str):
            mapping = {'vehicle': 0, 'bike': 1, 'pedestrian': 2}
            return mapping.get(obstacle_type_str.lower(), -1)

        # get the indices of the smallest k elements
        def get_indices_of_k_smallest(arr, k):
            idx = np.argpartition(arr, k)
            return idx[:k]
        distances = []

        vehicles_id = self.stub.GetVehicleIdList(
            trainsim_pb2.GetVehicleIdListReq(
                simulation_id=self.startResp.simulation_id, 
            ),
            metadata=self.metadata
        )
        vehicles_id_list = list(vehicles_id.list)[:100]
        if self.ego_id not in vehicles_id_list:
            vehicles_id_list[-1] = self.ego_id

        vehicles_position = self.stub.GetVehiclePosition(
            trainsim_pb2.GetVehiclePositionReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )

        vehicles_baseInfo = self.stub.GetVehicleBaseInfo(
            trainsim_pb2.GetVehicleBaseInfoReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )

        vehicles_MovingInfo = self.stub.GetVehicleMovingInfo(
            trainsim_pb2.GetVehicleMovingInfoReq(
                simulation_id=self.startResp.simulation_id, 
                id_list=vehicles_id_list
            ), 
            metadata=self.metadata
        )
        around_moving_objs = [id for id in vehicles_id_list if id != self.ego_id]

        # Preprocess the data of neighbor vehicles
        if (self.b_surr):
            if around_moving_objs == []:
                pass
            else:
                n = len(around_moving_objs)
                for i in range(n):
                    dist = CalDist(vehicles_position.position_dict.get(around_moving_objs[i]).point.x,
                                   vehicles_position.position_dict.get(around_moving_objs[i]).point.y, 
                                   vehicles_position.position_dict.get(self.ego_id).point.x,
                                   vehicles_position.position_dict.get(self.ego_id).point.y)
                    distances.append(dist)            
                # end_time = time.time()  # Start the timer
                # print(f'the for loop spend time {end_time - start_time} (s)')
            # sort out the nearest k distance vehicles
            if (len(distances) > self.surr_veh_num):
                indices = get_indices_of_k_smallest(distances, self.surr_veh_num)
            else:
                indices = range(len(distances))
            # append info of the nearest k distance vehicles
            for i in indices:
                # 濞ｈ濮炴稉鈧禍娑㈡绾板秶澧块悩鑸碘偓浣规殶閹癸拷
                obs_state = input_data.obs.obs_state.add()
                obs_state.x = vehicles_position.position_dict.get(around_moving_objs[i]).point.x
                obs_state.y = vehicles_position.position_dict.get(around_moving_objs[i]).point.y
                obs_state.phi = vehicles_position.position_dict.get(around_moving_objs[i]).phi
                obs_state.vx = math.sqrt(vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).u**2 
                                         + vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).v**2) * math.cos(obs_state.phi)

                obs_state.vy = math.sqrt(vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).u**2 
                                         + vehicles_MovingInfo.moving_info_dict.get(around_moving_objs[i]).v**2) * math.sin(obs_state.phi)
                obs_state.length = vehicles_baseInfo.info_dict.get(around_moving_objs[i]).base_info.length
                obs_state.width = vehicles_baseInfo.info_dict.get(around_moving_objs[i]).base_info.width

                obs_state.type = obstacle_type_to_int('vehicle')
                obs_state.id = id_to_numbers(around_moving_objs[i])
                # print(f'the obs_state:{obs_state.x}{obs_state.y}')

        input_data.ego.x = vehicles_position.position_dict.get(self.ego_id).point.x
        input_data.ego.y = vehicles_position.position_dict.get(self.ego_id).point.y
        input_data.ego.phi = vehicles_position.position_dict.get(self.ego_id).phi
        input_data.ego.vx = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).u
        input_data.ego.vy = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).v
        input_data.ego.r = vehicles_MovingInfo.moving_info_dict.get(self.ego_id).w
        input_data.ego.drive_mode = "auto"
        input_data.time_stamp.t = self.timestamp         
        self.timestamp += 1

    def covert_map(self, scenario_id):
        traffic_map = self.map_dict[scenario_id]
        for segment in traffic_map.data.segments:
            for link in segment.ordered_links:
                for lane in link.ordered_lanes:
                    print("lane: ",lane)
                    self.lanes[lane.id] = lane

    def get_direction(self, link_route: List[str]):
        direction_list = []
        assert len(link_route) > 0, "link_route is empty"
        for i in range(len(link_route) - 1):
            # 閼惧嘲褰囪ぐ鎾冲link閻ㄥ嫭澧嶉張澶夌瑓濞撶珮ink
            next_link_list = self.get_next_link(link_route[i])
            if link_route[i + 1] in next_link_list:
                direction_list.append(self.get_link_direction(link_route[i]))
            else:
                connection_id_list = self.get_connection_id(link_route[i], link_route[i + 1])
                if len(connection_id_list) == 0:
                    return "other"
                else:
                    direction_list.append(self.get_lane_direction(connection_id_list[0]))

        if "left" in direction_list and "right" in direction_list:
            return "other"
        elif "left" in direction_list:
            return "left"
        elif "right" in direction_list:
            return "right"
        else:
            return "straight"

    def get_next_link(self, link_id):
        link = self.links[link_id]
        lane_id_list = link.ordered_lane_ids
        next_link_set = set()

        for lane_id in lane_id_list:
            lane = self.lanes[lane_id]
            if not lane:
                continue

            next_link_id_list = lane.downstream_link_ids
            for next_link_id in next_link_id_list:
                next_link_set.add(next_link_id)

        return list(next_link_set)

    def get_link_direction(self, link_id):
        lane_id_list = self.links[link_id].ordered_lane_ids
        one_valid_lane_id = None
        for lane_id in lane_id_list:
            lane = self.lanes[lane_id]
            if (len(lane.center_line) > 0):
                one_valid_lane_id = lane_id
                break

        if one_valid_lane_id is None:
            raise ValueError("link_id `{}` has no path".format(link_id))

        return self.get_lane_direction(one_valid_lane_id)

    def get_lane_direction(self, lane_id):
        lane_path = self.get_ref_path_by_id(lane_id)
        if not lane_path:
            return "unknown"

        lane_path_phi = []
        for i in range(len(lane_path) - 1):
            lane_path_phi.append(math.atan2(lane_path[i + 1].y - lane_path[i].y, lane_path[i + 1].x - lane_path[i].x))
        lane_path_phi = np.array(lane_path_phi)
        # 鐠侊紕鐣婚懜顏勬倻鐟欐帒褰夐崠鏍电礉閼板啳妾荤搾濠呯箖 鍗よ熀 閻ㄥ嫭鍎忛崘锟�
        delta_phi = np.diff(lane_path_phi)
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # 鐏忓棜顫楁惔锕€妯婅ぐ鎺嶇閸栨牕鍩� [-锜�, 锜篯

        # 鐠侊紕鐣荤槐顖濐吀閻ㄥ嫯鍩呴崥鎴ｎ潡閸欐ê瀵�
        cumulative_phi_change = np.sum(delta_phi)

        # 閸掋倖鏌囩槐顖濐吀閸欐ê瀵查柌锟�
        if cumulative_phi_change > np.radians(30):
            return "left"
        elif cumulative_phi_change < np.radians(-30):
            return "right"
        elif cumulative_phi_change > np.radians(120):
            return "uturn"
        else:
            return "straight"

    def get_ref_path_by_id(self, lane_id):
        if lane_id in self.connections:
            return self.connections[lane_id].path
        else:
            return self.lanes[lane_id].center_line

    def get_connection_id(self, upstream_link_id, downstream_link_id):
        connection_id_list = []
        for key, connection in self.connections.items():
            if connection.upstream_link_id == upstream_link_id and connection.downstream_link_id == downstream_link_id:
                connection_id_list.append(connection.connection_id)
        return connection_id_list

    def convert_coordinates(self, x, y, source_origin, target_origin, angle):



        # 鐠侊紕鐣婚惄绋款嚠娴滃孩绨崸鎰垼缁甯悙鍦畱閸ф劖鐖ｉ崑蹇曅�
        relative_x = x - source_origin[0]
        relative_y = y - source_origin[1]

        rotated_x = relative_x * math.cos(angle) - relative_y * math.sin(angle)
        rotated_y = relative_x * math.sin(angle) + relative_y * math.cos(angle)

        target_x = rotated_x + target_origin[0]
        target_y = rotated_y + target_origin[1]

        return target_x, target_y

    def angle_normalize(self, x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def get_boundary_distance(self):
        def calculate_perpendicular_points(x0, y0, direction_radians, distance):
            dx = -math.sin(direction_radians)
            dy = math.cos(direction_radians)
            
            x1 = x0 + distance * dx
            y1 = y0 + distance * dy
            x2 = x0 - distance * dx
            y2 = y0 - distance * dy
            
            return (x1, y1), (x2, y2)
        def cal_distance(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        assert not self.in_junction
        pos = self.stub.GetVehiclePosition(
            trainsim_pb2.GetVehiclePositionReq(simulation_id=self.startResp.simulation_id,
                                               id_list=[self.ego_id]), metadata=self.metadata)
        ego_lane = pos.position_dict.get(self.ego_id).lane_id
        lane_list = self.lanes
        lane_width = 3.75
        target_lane = lane_list[ego_lane]
        left1,right1 = calculate_perpendicular_points(target_lane.center_line[0].point.x,target_lane.center_line[0].point.y,target_lane.center_line[0].heading,target_lane.center_line[0].left_width)
        left2,right2 = calculate_perpendicular_points(target_lane.center_line[1].point.x,target_lane.center_line[1].point.y,target_lane.center_line[1].heading,target_lane.center_line[1].left_width)

        left_lane_new,right_lane_new = [left1,left2],[right1,right2]
        
    
        left_lane = LineString(np.array(left_lane_new)[:, :2])
        right_lane = LineString(np.array(right_lane_new)[:, :2])
        Rb_position = point_project_to_line(right_lane, self._state[0], self._state[1])
        Rb_x, Rb_y, _ = compute_waypoint(right_lane, Rb_position)
        right_center_distance = cal_distance(Rb_x, Rb_y, self._state[0], self._state[1])

        Lb_position = point_project_to_line(left_lane, self._state[0], self._state[1])
        Lb_x, Lb_y, _ = compute_waypoint(left_lane, Lb_position)
        left_center_distance = cal_distance(Lb_x, Lb_y, self._state[0], self._state[1])
        return left_center_distance, right_center_distance

def env_creator(**kwargs: Any) -> LasvsimEnv:
    return LasvsimEnv(**kwargs)
