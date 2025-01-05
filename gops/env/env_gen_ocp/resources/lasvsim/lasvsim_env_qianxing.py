import random
from typing import Any, Dict, Tuple, List
import gym
import numpy as np
from gym.utils import seeding
import time
import math
import grpc
import time
import os
from shapely.geometry import Point, LineString, Polygon
from gops.utils.math_utils import deal_with_phi_rad, convert_ref_to_ego_coord
import matplotlib.pyplot as plt
from gops.utils.map_tool.utils import path_discrete_t_new
from risenlighten.lasvsim.train_sim.api.trainsim import trainsim_pb2
from risenlighten.lasvsim.train_sim.api.trainsim import trainsim_pb2_grpc
from risenlighten.lasvsim.train_sim.api.trainsim import scenario_pb2
from risenlighten.lasvsim.train_sim.api.trainsim import scenario_pb2_grpc
from risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1 import train_task_pb2
from risenlighten.lasvsim.lasvsim_web_bff.openapi.train_task.v1 import train_task_pb2_grpc

from collections import deque

import os
from gops.utils.map_tool.lib.map import Map
from gops.env.env_gen_ocp.resources.idsim_model.utils.las_render import \
    RenderCfg, _render_tags, LasStateSurrogate, append_to_pickle_incremental, render_tags_debug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gops.env.env_gen_ocp.resources.lib import point_project_to_line, \
    compute_waypoints_by_intervals, compute_waypoint, \
    create_box_polygon
from gops.env.env_gen_ocp.resources.lasvsim.log_utils import LoggingInterceptor
from gops.env.env_gen_ocp.resources.lasvsim.utils import \
    inverse_normalize_action, cal_dist, \
    get_indices_of_k_smallest, convert_ground_coord_to_ego_coord, \
    calculate_perpendicular_points, timeit

from dataclasses import dataclass


@dataclass(eq=False)
class EgoVehicle():
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    length: float = 0.0
    width: float = 0.0
    action: np.ndarray = np.array([0.0]*2)  # (real) acc, steer
    state: np.ndarray = np.array([0.0]*6)  # x, y, vx, vy, phi, w
    last_action: np.ndarray = np.array([0.0]*2)
    left_boundary_distance: float = 0.0
    right_boundary_distance: float = 0.0
    segment_id: str = 'default'
    junction_id: str = 'default'
    lane_id: str = 'default'
    link_id: str = 'default'
    in_junction: bool = False
    polygon: Polygon = None

    @property
    def ground_position(self) -> Tuple[float, float]:
        return self.state[0].item(), self.state[1].item()


@dataclass(eq=False)
class SurroundingVehicle():
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    rel_x: float = 0.0
    rel_y: float = 0.0
    rel_phi: float = 0.0
    u: float = 0.0
    distance: float = 0.0
    length: float = 0.0
    width: float = 0.0
    lane_id: str = 'default'
    mask: int = 0  # fake vehicle


@dataclass(eq=False)
class LasVSimContext():
    ego: EgoVehicle
    ref_list: List[LineString]
    sur_list: List[SurroundingVehicle]


class LasvsimEnv():
    def __init__(
        self,
        token: str,
        env_config: Dict = {},
        model_config: Dict = {},
        task_id=None,
        port: int = 8000,
        server_host='localhost:8290',
        render_info: dict = {},
        render_flag: bool = False,
        traj_flag: bool = False,
        **kwargs: Any,
    ):
        self.port = port
        self.metadata = [('authorization', 'Bearer ' + token)]
        assert task_id is not None, "None task id"

        # ================== 1. Build a connection ==================
        # connect train_task
        self.sce_insecure_channel = grpc.insecure_channel(server_host)
        self.sce_channel = grpc.intercept_channel(
            self.sce_insecure_channel.__enter__(), LoggingInterceptor())
        self.sce_stub = train_task_pb2_grpc.TrainTaskStub(self.sce_channel)

        # connect scenario
        self.insecure_channel = grpc.insecure_channel(server_host)
        self.channel = grpc.intercept_channel(
            self.insecure_channel.__enter__(), LoggingInterceptor())
        self.stub = trainsim_pb2_grpc.SimulationStub(self.channel)
        self.scenario_stub = scenario_pb2_grpc.ScenarioStub(self.channel)

        # get scene list and init one
        scene_id_info = self.sce_stub.GetSceneIdList(train_task_pb2.GetSceneIdListRequest(task_id=task_id),
                                                     metadata=self.metadata)
        self.scenario_list = scene_id_info.scene_id_list
        self.version_list = scene_id_info.scene_version_list
        self.scenario_id = self.scenario_list[0]

        try:
            self.startResp = self.init_remote_lasvsim(
                scenario_id=self.scenario_list[0],
                scenario_version=self.version_list[0]
            )
        except grpc.RpcError as e:
            print(f"gRPC error during initialization: {e}")
            raise

        # ================== 2. Process static map and render ==================
        self.connections = {}
        self.junctions = {}
        self.lanes = {}
        self.segments = {}
        self.links = {}
        self.map_dict = {}
        for i in range(len(self.scenario_list)):
            startResp = self.init_remote_lasvsim(
                scenario_id=self.scenario_list[i],
                scenario_version=self.version_list[i]
            )
            cur_map = self.get_remote_hdmap(startResp)
            self.map_dict[self.scenario_list[i]] = cur_map

        self.covert_map(self.scenario_list[0])
        

        # ================== 3. Init simulator ==================
        # init variables
        self.config = env_config
        self.pre_horizon = model_config["N"]
        self.scenario_cnt = 0
        self.timestamp = 0
        self.ref_index = 0
        self.action_lower_bound = np.array(self.config["action_lower_bound"])
        self.action_upper_bound = np.array(self.config["action_upper_bound"])
        self.action_center = (self.action_upper_bound +
                              self.action_lower_bound) / 2
        self.action_half_range = (
            self.action_upper_bound - self.action_lower_bound) / 2
        self.real_action_upper = np.array(
            self.config["real_action_upper_bound"])
        self.real_action_lower = np.array(
            self.config["real_action_lower_bound"])
        self.surr_veh_num = self.config['obs_num_surrounding_vehicles']['passenger']
            # self.config['obs_num_surrounding_vehicles']['bicycle'] +\
            # self.config['obs_num_surrounding_vehicles']['pedestrian']
        
        # init ego vehicle
        test_vehicle = self.get_remote_lasvsim_test_veh_list()
        self.ego_id = random.choice(test_vehicle.list)
        self.lasvsim_context = LasVSimContext(
            ego=EgoVehicle(),
            ref_list=[],
            sur_list=[]
        )
        self.step_remote_lasvsim()
        self.update_lasvsim_context()
        self.render_flag = render_flag
        self.traj_flag = traj_flag  # for log
        self._render_init(render_info=render_info)

    def init_remote_lasvsim(self, scenario_id: str, scenario_version: str):
        startResp = self.stub.Init(
            trainsim_pb2.InitReq(scenario_id=scenario_id,
                                 scenario_version=scenario_version),
            metadata=self.metadata
        )
        return startResp

    def reset_remote_lasvsim(self):
        resetResp = self.stub.Reset(
            trainsim_pb2.ResetReq(simulation_id=self.startResp.simulation_id,
                                  scenario_id=self.scenario_id),
            metadata=self.metadata,
        )
        return resetResp

    def step_remote_lasvsim(self):
        stepResult = self.stub.Step(
            trainsim_pb2.StepReq(simulation_id=self.startResp.simulation_id
                                 ),
            metadata=self.metadata
        )
        return stepResult

    def stop_remote_lasvsim(self, simulation_id: str = None):
        if simulation_id is None:
            simulation_id = self.startResp.simulation_id
        resetResp = self.stub.Stop(
            trainsim_pb2.StopReq(simulation_id=simulation_id),
            metadata=self.metadata,
        )
        return resetResp

    def update_lasvsim_context(self, real_action: np.ndarray = None):
        ego_context = self.get_ego_context(real_action)
        ref_contex = self.get_ref_context()
        sur_context = self.get_sur_context()
        self.lasvsim_context = LasVSimContext(
            ego=ego_context,
            ref_list=ref_contex,
            sur_list=sur_context
        )

    def step(self, action: np.ndarray):
        # action: network output, \in [-1, 1]
        action = inverse_normalize_action(action, self.action_half_range, self.action_center)
        real_action = action + self.lasvsim_context.ego.last_action
        real_action = np.clip(
            real_action, self.real_action_lower, self.real_action_upper)
        self.set_remote_lasvsim_veh_control(real_action)
        self.step_remote_lasvsim()
        self.update_lasvsim_context(real_action)

        reward, rew_info = self.reward_function_multilane()

        obs = None
        return obs, reward, self.judge_done(), self.judge_done(), rew_info

    def reset(self):
        test_vehicle_list = []
        if self.scenario_cnt < 10:
            while len(test_vehicle_list) == 0:
                self.reset_remote_lasvsim()
                self.step_remote_lasvsim()
                test_vehicle = self.get_remote_lasvsim_test_veh_list()
                if test_vehicle is not None:
                    test_vehicle_list = test_vehicle.list
            self.scenario_cnt += 1
        else:
            while len(test_vehicle_list) == 0:
                self.stop_remote_lasvsim()
                idx = random.randint(0, len(self.scenario_list) - 1)
                self.scenario_id = self.scenario_list[idx]
                self.startResp = self.init_remote_lasvsim(
                    scenario_id=self.scenario_id,
                    scenario_version=self.version_list[idx]
                )
                self.step_remote_lasvsim()
                test_vehicle = self.get_remote_lasvsim_test_veh_list()
                if (test_vehicle is not None):
                    test_vehicle_list = test_vehicle.list
                self.covert_map(self.scenario_list[idx])
            self.scenario_cnt = 0

        self.ego_id = test_vehicle_list[0]
        self.update_lasvsim_context()
        obs = None
        info = {}
        return obs, info

    # from rlplanner
    def model_free_reward_multilane_batch(self,
                                          context,  # S_t, model_context
                                          # absolute action, A_{t-2}
                                          last_last_action,
                                          # absolute action, A_{t-1}
                                          last_action,
                                          # normalized incremental action, （A_t - A_{t-1}�? / Z
                                          action
                                          ) -> Tuple[List[np.ndarray], List[dict]]:
        # all inputs are batched
        # vehicle state: context.x.ego_state
        ego_state = context.x.ego_state[0]  # [6]: x, y, vx, vy, phi, r
        # [R, 2N+1, 4] ref_x, ref_y, ref_phi, ref_v
        ref_param = context.p.ref_param[0]
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state
        last_acc, last_steer = last_action[0][0], last_action[0][1] * 180 / np.pi
        last_last_acc, last_last_steer = last_last_action[0][0], last_last_action[0][1] * 180 / np.pi
        delta_steer = (last_steer - last_last_steer) / self.config["dt"]
        jerk = (last_acc - last_last_acc) / self.config["dt"]

        ref_states = ref_param[:, context.i, :]  # [R, 4]
        next_ref_states = ref_param[:, context.i + 1, :]  # [R, 4]
        ref_x, ref_y, ref_phi, ref_v = ref_states.T
        next_ref_v = next_ref_states[:, 3]

        # live reward
        rew_step = 0.5 * np.clip(ego_vx, 0, 2.0) * \
            np.ones(ref_param.shape[0])  # 0~1

        # tracking_error
        tracking_error = -(ego_x - ref_x) * np.sin(ref_phi) + \
            (ego_y - ref_y) * np.cos(ref_phi)
        delta_phi = deal_with_phi_rad(
            ego_phi - ref_phi) * 180 / np.pi  # degree
        ego_r = ego_r * 180 / np.pi  # degree
        speed_error = ego_vx - ref_v

        # tracking_error
        punish_dist_lat = 5 * np.where(
            np.abs(tracking_error) < 0.3,
            np.square(tracking_error),
            0.02 * np.abs(tracking_error) + 0.084,
        )  # 0~1 0~6m 50% 0~0.3m

        punish_vel_long = 0.5*np.where(
            np.abs(speed_error) < 1,
            np.square(speed_error),
            0.2*np.abs(speed_error)+0.8,
        )  # 0~1 0~11.5m/s 50% 0~1m/s

        punish_head_ang = 0.05 * np.where(
            np.abs(delta_phi) < 3,
            np.square(delta_phi),
            np.abs(delta_phi) + 8,
        )  # 0~1  0~12 degree 50% 0~3 degree

        ego_r = ego_r * np.ones(ref_param.shape[0])
        punish_yaw_rate = 0.1 * np.where(
            np.abs(ego_r) < 2,
            np.square(ego_r),
            np.abs(ego_r) + 2,
        )  # 0~1  0~8 degree/s 50% 0~2 degree/s

        punish_overspeed = np.clip(
            np.where(
                ego_vx > 1.05 * ref_v,
                1 + np.abs(ego_vx - 1.05 * ref_v),
                0, ),
            0, 2)

        # reward related to action
        nominal_steer = self._get_nominal_steer_by_state_batch(
            ego_state, ref_param) * 180 / np.pi

        abs_steer = np.abs(last_steer - nominal_steer)
        reward_steering = -np.where(abs_steer < 4,
                                    np.square(abs_steer), 2 * abs_steer + 8)

        self.out_of_action_range = abs_steer > 20

        if ego_vx < 0.1 and self.config["enable_slow_reward"]:
            reward_steering = reward_steering * 5

        abs_ax = np.abs(last_acc) * np.ones(ref_param.shape[0])
        reward_acc_long = -np.where(abs_ax < 2, np.square(abs_ax), 2 * abs_ax)

        delta_steer = delta_steer * np.ones(ref_param.shape[0])
        reward_delta_steer = - \
            np.where(np.abs(delta_steer) < 4, np.square(
                delta_steer), 2 * np.abs(delta_steer) + 8)
        jerk = jerk * np.ones(ref_param.shape[0])
        reward_jerk = -np.where(np.abs(jerk) < 2,
                                np.square(jerk), 2 * np.abs(jerk) + 8)

        # if self.in_multilane:  # consider more comfortable reward
        #     reward_acc_long = reward_acc_long * 2
        #     reward_jerk = reward_jerk * 2
        #     reward_steering = reward_steering * 2
        #     reward_delta_steer = reward_delta_steer * 2
        #     punish_yaw_rate = punish_yaw_rate * 2

        # if self.turning_direction != 0:  # left is positive =1
        #     punish_dist_lat = punish_dist_lat * 0.5
        #     punish_head_ang = punish_head_ang * 0.5
        #     punish_yaw_rate = punish_yaw_rate * 0.2
        #     reward_steering = reward_steering * 0.2
        #     tracking_bias_direrction = np.sign(
        #         tracking_error)  # left is positive
        #     phi_direrction = np.sign(delta_phi)  # left is positive
        #     condition = (self.turning_direction != tracking_bias_direrction) & (
        #         self.turning_direction != phi_direrction) & (np.abs(tracking_error) > 0.05) & (np.abs(delta_phi) > 2)
        #     punish_dist_lat = np.where(
        #         condition, punish_dist_lat + 4, punish_dist_lat)
        #     punish_head_ang = np.where(
        #         condition, punish_head_ang + 4, punish_head_ang)

        break_condition = (ref_v < 1.5) & (
            (next_ref_v - ref_v) < -0.1) | (ref_v < 1.0)
        if break_condition.any() and self.config["nonimal_acc"]:
            nominal_acc = np.where(break_condition, -1.5, 0)
            # remove the effect of tracking error
            punish_dist_lat = np.where(break_condition, 0, punish_dist_lat)
            punish_head_ang = np.where(break_condition, 0, punish_head_ang)
            reward_acc_long = np.where(break_condition, 0, reward_acc_long)
        else:
            nominal_acc = np.zeros(ref_param.shape[0])
            punish_nominal_acc = np.zeros(ref_param.shape[0])

        if self.braking_mode and self.config["nonimal_acc"]:
            nominal_acc = -1.5 * np.ones(ref_param.shape[0])
            punish_vel_long = np.zeros(ref_param.shape[0])

        if break_condition.any() or self.braking_mode:
            rew_step = np.where(break_condition, 1.0, rew_step)

        delta_acc = np.abs(nominal_acc - last_acc)
        punish_nominal_acc = (nominal_acc != 0) * np.where(delta_acc <
                                                           0.5, np.square(delta_acc), delta_acc - 0.25)

        # tracking related reward
        scaled_punish_dist_lat = punish_dist_lat * self.config["P_lat"]
        scaled_punish_vel_long = punish_vel_long * self.config["P_long"]
        scaled_punish_head_ang = punish_head_ang * self.config["P_phi"]
        scaled_punish_yaw_rate = punish_yaw_rate * self.config["P_yaw"]
        scaled_punish_overspeed = punish_overspeed * 3  # TODO: hard coded value

        # action related reward
        scaled_reward_steering = reward_steering * self.config["P_steer"]
        scaled_reward_acc_long = reward_acc_long * self.config["P_acc"]
        scaled_reward_delta_steer = reward_delta_steer * self.config["P_delta_steer"]
        scaled_reward_jerk = reward_jerk * self.config["P_jerk"]
        scaled_punish_nominal_acc = punish_nominal_acc * 8  # TODO: hard coded value

        # live reward
        scaled_rew_step = rew_step * self.config["R_step"]

        reward_ego_state = scaled_rew_step - \
            (scaled_punish_dist_lat +
             scaled_punish_vel_long +
             scaled_punish_head_ang +
             scaled_punish_yaw_rate +
             scaled_punish_nominal_acc +
             scaled_punish_overspeed) + \
            (scaled_reward_steering +
             scaled_reward_acc_long +
             scaled_reward_delta_steer +
             scaled_reward_jerk)

        reward_ego_state = np.clip(reward_ego_state, -5, 30)

        rewards = reward_ego_state.tolist()
        infos = [{
            "env_tracking_error": np.abs(tracking_error[i]),
            "env_speed_error": np.abs(speed_error[i]),
            "env_delta_phi": np.abs(delta_phi[i]),
            "state_nominal_steer": nominal_steer[i],
            "state_nominal_acc": nominal_acc[i],

            "env_reward_step": rew_step[i],

            "env_reward_steering": reward_steering[i],
            "env_reward_acc_long": reward_acc_long[i],
            "env_reward_delta_steer": reward_delta_steer[i],
            "env_reward_jerk": reward_jerk[i],

            "env_reward_dist_lat": -punish_dist_lat[i],
            "env_reward_vel_long": -punish_vel_long[i],
            "env_reward_head_ang": -punish_head_ang[i],
            "env_reward_yaw_rate": -punish_yaw_rate[i],

            "env_scaled_reward_part2": reward_ego_state[i],
            "env_scaled_reward_step": scaled_rew_step[i],
            "env_scaled_reward_dist_lat": -scaled_punish_dist_lat[i],
            "env_scaled_reward_vel_long": -scaled_punish_vel_long[i],
            "env_scaled_reward_head_ang": -scaled_punish_head_ang[i],
            "env_scaled_reward_yaw_rate": -scaled_punish_yaw_rate[i],
            "env_scaled_reward_nominal_acc": -scaled_punish_nominal_acc[i],
            "env_scaled_reward_overspeed": -scaled_punish_overspeed[i],
            "env_scaled_reward_steering": scaled_reward_steering[i],
            "env_scaled_reward_acc_long": scaled_reward_acc_long[i],
            "env_scaled_reward_delta_steer": scaled_reward_delta_steer[i],
            "env_scaled_reward_jerk": scaled_reward_jerk[i],
        } for i in range(ref_param.shape[0])]

        return rewards, infos

    def _get_nominal_steer_by_state_batch(self,
                                          ego_state,
                                          ref_param):
        # ref_param: [R, 2N+1, 4]
        # use ref_state_index to determine the start, from 2N+1 to 3
        # ref_line: [R, 3, 4]
        def cal_curvature(x1, y1, x2, y2, x3, y3):
            # cal curvature by three points in batch format
            # dim of x1 is [R]
            a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
            k = np.zeros_like(a)
            i = (a * b * c) != 0
            area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            k[i] = 2 * area[i] / (a[i] * b[i] * c[i])
            return k

        ref_line = np.stack([ref_param[:, i, :]
                            for i in [0, 5, 10]], axis=1)  # [R, 3, 4]
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = \
            convert_ref_to_ego_coord(ref_line[:, :, :3], ego_state)  # [R, 3]

        # nominal action
        x1, y1 = ref_x_ego_coord[:, 0], ref_y_ego_coord[:, 0]  # [R,]
        x2, y2 = ref_x_ego_coord[:, 1], ref_y_ego_coord[:, 1]
        x3, y3 = ref_x_ego_coord[:, 2], ref_y_ego_coord[:, 2]
        nominal_curvature = cal_curvature(x1, y1, x2, y2, x3, y3)
        nominal_steer = nominal_curvature * 2.65  # FIXME: hard-coded: wheel base
        nominal_steer = np.clip(
            nominal_steer, self.real_action_lower[1], self.real_action_upper[1])

        return nominal_steer

    # from rlplanner
    def reward_function_multilane(self):
        ego = self.lasvsim_context.ego
        # cal reference_closest
        ref_list = self.lasvsim_context.ref_list
        # try:
        closest_idx = np.argmin([ref_line.distance(ego.polygon)
                                    for ref_line in ref_list])
        # except Exception as e:
            # breakpoint()
        reference_closest = ref_list[closest_idx]
        # tracking_error cost
        position_on_ref = point_project_to_line(
            reference_closest, ego.x, ego.y)
        current_first_ref_x, current_first_ref_y, \
            current_first_ref_phi = compute_waypoint(
                reference_closest, position_on_ref)

        tracking_error = np.sqrt((ego.x - current_first_ref_x) ** 2 +
                                 (ego.y - current_first_ref_y) ** 2)
        delta_phi = deal_with_phi_rad(ego.phi - current_first_ref_phi)

        self.out_of_range = tracking_error > 4 or np.abs(delta_phi) > np.pi/4
        self.in_junction = ego.in_junction
        # self.in_multilane = self.engine.context.scenario_id in self.config.multilane_scenarios  # FIXME: hardcoded scenario_id
        # direction = vehicle.direction
        # if self.in_junction:
        #     self.turning_direction = 1 if direction == "l" else -1 if direction == "r" else 0
        # else:
        #     self.turning_direction = 0

        # TODO: hard coded value

        # ax = vehicle.ax

        # collision risk cost
        # ego_vx = vehicle.vx
        # ego_W = vehicle.width
        # ego_L = vehicle.length

        safety_lat_margin_front = self.config["safety_lat_margin_front"]
        safety_lat_margin_rear = safety_lat_margin_front  # TODO: safety_lat_margin_rear
        safety_long_margin_front = self.config["safety_long_margin_front"]
        safety_long_margin_side = self.config["safety_long_margin_side"]
        front_dist_thd = self.config["front_dist_thd"]
        space_dist_thd = self.config["space_dist_thd"]
        rel_v_thd = self.config["rel_v_thd"]
        rel_v_rear_thd = self.config["rel_v_rear_thd"]
        time_dist = self.config["time_dist"]

        punish_done = self.config["P_done"]

        pun2front = 0.
        pun2side = 0.
        pun2space = 0.
        pun2rear = 0.

        pun2front_sum = 0.
        pun2side_sum = 0.
        pun2space_sum = 0.
        pun2rear_sum = 0.

        min_front_dist = np.inf

        sur_list: List[SurroundingVehicle] = self.lasvsim_context.sur_list
        # sur_info = self.engine.context.vehicle.surrounding_veh_info
        # ego_edge = self.engine.context.vehicle.edge
        # ego_lane = self.engine.context.vehicle.lane
        # if self.config.ignore_opposite_direction and self.engine.context.scenario_id in self.config.multilane_scenarios:  # FIXME: hardcoded scenario_id
        #     sur_info = [s for s in sur_info if s.road_id == ego_edge]

        ego_W = ego.width
        ego_L = ego.length
        ego_vx = ego.u
        for sur_vehicle in sur_list:
            rel_x = sur_vehicle.rel_x
            rel_y = sur_vehicle.rel_y
            sur_vx = sur_vehicle.u
            sur_lane = sur_vehicle.lane_id
            sur_W = sur_vehicle.width
            sur_L = sur_vehicle.length
            # [1 - tanh(x)]: 0.25-> 75%  0.5->54%, 1->24%, 1.5->9.5% 2->3.6%, 3->0.5%
            if ((np.abs(rel_y) < (ego_W + sur_W) / 2 - 1)) \
                    and (rel_x > (ego_L + sur_L) / 2):
                min_front_dist = min(
                    min_front_dist, rel_x - (ego_L + sur_L) / 2)

            pun2front_cur = np.where(
                (np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front) and
                (rel_x >= 0) and (rel_x < front_dist_thd) and (ego_vx > sur_vx),
                np.clip(1. - np.tanh((rel_x-(ego_L + sur_L) / 2
                        - safety_long_margin_front) / (time_dist*(np.max(ego_vx, 0) + 0.1))),
                        0., 1.),
                0,
            )
            pun2front = np.maximum(pun2front, pun2front_cur)
            pun2front_sum += pun2front_cur

            pun2side_cur = np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 + safety_long_margin_side and rel_y *
                delta_phi > 0 and rel_y > (ego_W + sur_W) / 2,
                np.clip(1. - np.tanh((np.abs(rel_y) - (ego_W + sur_W) / 2) /
                        (np.abs(ego_vx*np.sin(delta_phi))+0.01)), 0., 1.),
                0,
            )
            pun2side = np.maximum(pun2side, pun2side_cur)
            pun2side_sum += pun2side_cur

            pun2space_cur = np.where(
                np.abs(rel_y) < (ego_W + sur_W) /
                2 and rel_x >= 0 and rel_x < space_dist_thd and ego_vx > sur_vx + rel_v_thd,
                np.clip(1. - (rel_x - (ego_L + sur_L) / 2) /
                        (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,) + np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 +
                safety_long_margin_side and np.abs(
                    rel_y) > (ego_W + sur_W) / 2,
                np.clip(
                    1. - np.tanh(3.0*(np.abs(rel_y) - (ego_W + sur_W) / 2)), 0., 1.),
                0,)
            pun2space = np.maximum(pun2space, pun2space_cur)
            pun2space_sum += pun2space_cur

            pun2rear_cur = np.where(
                (np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_rear) and rel_x < 0 and rel_x > -
                space_dist_thd and ego_vx < sur_vx - rel_v_rear_thd,
                np.clip(1. - (-1)*(rel_x + (ego_L + sur_L) / 2) /
                        (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,)
            pun2rear = np.maximum(pun2rear, pun2rear_cur)
            pun2rear_sum += pun2rear_cur

        if self.config["punish_sur_mode"] == "sum":
            pun2front = pun2front_sum
            pun2side = pun2side_sum
            pun2space = pun2space_sum
            pun2rear = pun2rear_sum
        elif self.config["punish_sur_mode"] == "max":
            pass
        else:
            print(self.config["punish_sur_mode"])
            raise ValueError(
                f"Invalid punish_sur_mode")
        scaled_pun2front = pun2front * self.config["P_front"]
        scaled_pun2side = pun2side * self.config["P_side"]
        scaled_pun2space = pun2space * self.config["P_space"]
        scaled_pun2rear = pun2rear * self.config["P_rear"]

        # self.braking_mode = (
        #     min_front_dist < 4) and not self.in_junction and not self.in_multilane  # trick
        self.braking_mode = False

        punish_collision_risk = scaled_pun2front + \
            scaled_pun2side + scaled_pun2space + scaled_pun2rear

        if ego_vx <= 0.01:
            punish_collision_risk = 0

        # exclude scenarios without surrounding vehicles
        self.active_collision = self.check_collision() and ego_vx > 0.01

        # out of driving area cost
        # TODO: boundary cost = 0  when boundary info is not available
        if self.in_junction or self.config["P_boundary"] == 0:
            punish_boundary = 0.
        else:
            rel_angle = np.abs(delta_phi)
            left_distance = np.abs(ego.left_boundary_distance)
            right_distance = np.abs(ego.right_boundary_distance)
            min_left_distance = left_distance - \
                (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
            min_right_distance = right_distance - \
                (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
            boundary_safe_margin = 0.15
            boundary_distance = np.clip(np.minimum(
                min_left_distance, min_right_distance), 0., None)

            punish_boundary = np.where(
                boundary_distance < boundary_safe_margin,
                np.clip((1. - boundary_distance/boundary_safe_margin), 0., 1.),
                0.0,
            )
        scaled_punish_boundary = punish_boundary * self.config["P_boundary"]

        # action related reward

        reward = - scaled_punish_boundary

        punish_collision_risk = punish_collision_risk if (
            self.config["penalize_collision"]) else 0.
        reward -= punish_collision_risk

        event_flag = 0  # nomal driving (on lane, stop)
        reward_done = 0
        reward_collision = 0
        # Event reward: target reached, collision, out of driving area
        if self.check_out_of_driving_area() or self.out_of_range:  # out of driving area
            reward_done = - punish_done
            event_flag = 1
        elif self.active_collision:  # collision by ego vehicle
            reward_collision = -20 if self.config["penalize_collision"] else 0.
            event_flag = 2
        elif self.braking_mode:  # start to brake
            event_flag = 3

        reward += (reward_done + reward_collision)

        # if vehicle.arrive_success:
        #     reward += 200.
        return reward, {
            "category": event_flag,
            "env_pun2front": pun2front,
            "env_pun2side": pun2side,
            "env_pun2space": pun2space,
            "env_pun2rear": pun2rear,
            "env_scaled_reward_part1": reward,
            "env_scaled_reward_done": reward_done,
            "env_scaled_reward_collision": reward_collision,
            "env_scaled_reward_collision_risk": - punish_collision_risk,
            "env_scaled_pun2front": scaled_pun2front,
            "env_scaled_pun2side": scaled_pun2side,
            "env_scaled_pun2space": scaled_pun2space,
            "env_scaled_pun2rear": scaled_pun2rear,
            "env_scaled_reward_boundary": - scaled_punish_boundary,
        }
    
    def check_collision(self) -> bool:
        collision_info = self.stub.GetVehicleCollisionInfo(
            trainsim_pb2.GetVehicleCollisionInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id=self.ego_id,
            ),
            metadata=self.metadata
        )
        collision_flag = collision_info.collision_flag
        return collision_flag

    def check_out_of_driving_area(self) -> bool:
        vehicles_position = self.get_remote_lasvsim_veh_position()
        ego_pos = vehicles_position.position_dict.get(self.ego_id).position_type
        out_of_driving_area_flag = (ego_pos == 1)
        return out_of_driving_area_flag

    def judge_done(self) -> bool:
        collision = self.check_collision()
        out_of_driving_area = self.check_out_of_driving_area()
        park_flag = (self.lasvsim_context.ego.u == 0)
        out_of_defined_region = self.out_of_range
        self._render_done_info = {
            "Pause": park_flag,
            "RegionOut": out_of_defined_region,
            "Collision": collision,
            "MapOut": out_of_driving_area
        }
        done = collision or out_of_defined_region or out_of_driving_area
        return done

    def get_ego_context(self, real_actiton: np.ndarray = None):
        vehicles_position = self.get_remote_lasvsim_veh_position()
        vehicles_baseInfo = self.get_remote_lasvsim_veh_base_info()
        vehicles_movingInfo = self.get_remote_lasvsim_veh_moving_info()

        x = vehicles_position.position_dict.get(self.ego_id).point.x
        y = vehicles_position.position_dict.get(self.ego_id).point.y
        phi = vehicles_position.position_dict.get(self.ego_id).phi
        junction_id = vehicles_position.position_dict.get(
            self.ego_id).junction_id
        lane_id = vehicles_position.position_dict.get(self.ego_id).lane_id
        link_id = vehicles_position.position_dict.get(self.ego_id).link_id
        segment_id = vehicles_position.position_dict.get(
            self.ego_id).segment_id
        ego_pos = vehicles_position.position_dict.get(
            self.ego_id).position_type
        in_junction = (ego_pos == 2)

        length = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.length
        width = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.width

        u = vehicles_movingInfo.moving_info_dict.get(self.ego_id).u
        v = vehicles_movingInfo.moving_info_dict.get(self.ego_id).v
        w = vehicles_movingInfo.moving_info_dict.get(self.ego_id).w

        assert not in_junction
        can_not_get_lane_id = False
        try:
            target_lane = self.lanes[lane_id]
        except Exception as e:
            print('X'*50)
            print('can_not_get_lane_id')
            can_not_get_lane_id = True

        if can_not_get_lane_id:
            right_boundary_distance = 1.5
            left_boundary_distance = 1.5
        else:
            # print("Normal lane id")
            left1, right1 = calculate_perpendicular_points(
                target_lane.center_line[0].point.x,
                target_lane.center_line[0].point.y,
                target_lane.center_line[0].heading,
                target_lane.center_line[0].left_width)
            left2, right2 = calculate_perpendicular_points(
                target_lane.center_line[1].point.x,
                target_lane.center_line[1].point.y,
                target_lane.center_line[1].heading,
                target_lane.center_line[1].left_width)
            # print("center1: ", target_lane.center_line[0])
            # print("center2: ", target_lane.center_line[1])
            # print("boundary points: ", left1, right1)
            # print("boundary points: ", left2, right2)
            # breakpoint()

            left_boundary_lane = LineString([left1, left2])
            right_boundary_lane = LineString([right1, right2])
            Rb_position = point_project_to_line(right_boundary_lane, x, y)
            Rb_x, Rb_y, _ = compute_waypoint(right_boundary_lane, Rb_position)
            right_boundary_distance = cal_dist(Rb_x, Rb_y, x, y)

            Lb_position = point_project_to_line(left_boundary_lane, x, y)
            Lb_x, Lb_y, _ = compute_waypoint(left_boundary_lane, Lb_position)
            left_boundary_distance = cal_dist(Lb_x, Lb_y, x, y)

        polygon = create_box_polygon(x, y, phi, length, width)

        if real_actiton is not None:
            action = real_actiton
        else:
            action = np.array([0.0]*2)
        last_action = self.lasvsim_context.ego.action

        return EgoVehicle(
            x=x, y=y, phi=phi, u=u, v=v, w=w,
            length=length, width=width,
            action=action,
            last_action=last_action,
            junction_id=junction_id, lane_id=lane_id,
            link_id=link_id, segment_id=segment_id,
            in_junction=in_junction,
            left_boundary_distance=left_boundary_distance,
            right_boundary_distance=right_boundary_distance,
            polygon=polygon
        )

    def get_ref_context(self):
        ref_points = self.get_remote_lasvsim_ref_line()
        # breakpoint()
        ref_lines = ref_points.reference_lines
        if len(ref_lines)==0:
            print('X'*50)
            print("Zero ref!!!")
            return self.lasvsim_context.ref_list
        # print("Normal ref")
        ref_context = []
        # assert len(ref_lines) > 0
        # assert len(ref_lines[0].points) > 1
        for ref_line in ref_lines:
            ref_line_xy = np.array([[point.x, point.y]
                                for point in ref_line.points])
            ref_line_string = LineString(ref_line_xy)
            ref_context.append(ref_line_string)
        return ref_context

    def get_sur_context(self):
        perception_info = self.get_remote_lasvsim_perception_info()
        around_moving_objs = perception_info.list
        self._render_parse_surcar(around_moving_objs)

        ego_x, ego_y, ego_phi = self.lasvsim_context.ego.x, \
            self.lasvsim_context.ego.y, \
            self.lasvsim_context.ego.phi

        # filter neighbor vehicles for better efficiency
        distances = [
            cal_dist(
                obj.position.point.x,
                obj.position.point.y,
                ego_x,
                ego_y
            )
            for obj in around_moving_objs
        ]

        # sort out the smallest k distance vehicles
        if (len(distances) > self.surr_veh_num):
            indices = get_indices_of_k_smallest(distances, self.surr_veh_num)
        else:
            indices = range(len(distances))

        # append info of the smallest k distance vehicles
        sur_context = []
        for i in indices:
            sur_x, sur_y, sur_phi = \
                around_moving_objs[i].position.point.x, \
                around_moving_objs[i].position.point.y, \
                around_moving_objs[i].position.phi
            rel_x, rel_y, rel_phi = convert_ground_coord_to_ego_coord(
                sur_x, sur_y, sur_phi,
                ego_x, ego_y, ego_phi
            )
            distance = distances[i]
            u = around_moving_objs[i].moving_info.u
            length = around_moving_objs[i].base_info.length
            width = around_moving_objs[i].base_info.width
            lane_id = around_moving_objs[i].position.lane_id
            sur_vehicle = SurroundingVehicle(
                x=sur_x, y=sur_y, phi=sur_phi,
                rel_x=rel_x, rel_y=rel_y, rel_phi=rel_phi,
                u=u, distance=distance,
                length=length, width=width,
                lane_id=lane_id,
                mask=1
            )
            if rel_phi < np.pi/2 and distance > 0.01:
                sur_context.append(sur_vehicle)

        sur_context.extend(SurroundingVehicle()
                           for _ in range(self.surr_veh_num - len(sur_context)))
        return sur_context

    def covert_map(self, scenario_id: str):
        traffic_map = self.map_dict[scenario_id]
        for segment in traffic_map.data.segments:
            for link in segment.ordered_links:
                for lane in link.ordered_lanes:
                    # print("lane: ", lane)
                    self.lanes[lane.id] = lane

    def get_remote_hdmap(self, startResp):
        hdmap = self.scenario_stub.GetHdMap(
            scenario_pb2.GetHdMapReq(simulation_id=startResp.simulation_id),
            metadata=self.metadata
        )
        return hdmap

    def get_remote_lasvsim_test_veh_list(self):
        test_vehicle = self.stub.GetTestVehicleIdList(
            trainsim_pb2.GetTestVehicleIdListReq(
                simulation_id=self.startResp.simulation_id,
            ),
            metadata=self.metadata
        )
        return test_vehicle

    def set_remote_lasvsim_veh_control(self, real_action: np.ndarray):
        lon_acc, ste_wheel = real_action
        vehicleControlResult = self.stub.SetVehicleControlInfo(
            trainsim_pb2.SetVehicleControlInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id=self.ego_id,
                lon_acc=lon_acc,
                ste_wheel=ste_wheel
            ),
            metadata=self.metadata
        )
        return vehicleControlResult

    def get_remote_lasvsim_veh_position(self):
        vehicles_position = self.stub.GetVehiclePosition(
            trainsim_pb2.GetVehiclePositionReq(
                simulation_id=self.startResp.simulation_id,
                id_list=[self.ego_id]
            ),
            metadata=self.metadata
        )
        return vehicles_position

    def get_remote_lasvsim_veh_base_info(self):
        vehicles_baseInfo = self.stub.GetVehicleBaseInfo(
            trainsim_pb2.GetVehicleBaseInfoReq(
                simulation_id=self.startResp.simulation_id,
                id_list=[self.ego_id]
            ),
            metadata=self.metadata
        )
        return vehicles_baseInfo

    def get_remote_lasvsim_veh_moving_info(self):
        vehicles_MovingInfo = self.stub.GetVehicleMovingInfo(
            trainsim_pb2.GetVehicleMovingInfoReq(
                simulation_id=self.startResp.simulation_id,
                id_list=[self.ego_id]
            ),
            metadata=self.metadata
        )
        return vehicles_MovingInfo

    def get_remote_lasvsim_ref_line(self):
        ref_points = self.stub.GetVehicleReferenceLines(
            trainsim_pb2.GetVehicleReferenceLinesReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id=self.ego_id
            ),
            metadata=self.metadata
        )
        return ref_points

    def get_remote_lasvsim_perception_info(self):
        perception_info = self.stub.GetVehiclePerceptionInfo(
            trainsim_pb2.GetVehiclePerceptionInfoReq(
                simulation_id=self.startResp.simulation_id,
                vehicle_id=self.ego_id,
            ),
            metadata=self.metadata
        )
        return perception_info

    def __del__(self):
        self.insecure_channel.__exit__(None, None, None)

    # ================== Render utils ==================

    # Core params
    _render_count = 0
    _render_tags = _render_tags
    _render_tags_debug = render_tags_debug
    _render_cfg: RenderCfg
    render_flag: bool
    traj_flag: bool

    # data buffers
    _render_info = {}
    _render_ego_shadows = deque([])
    _render_surcars: list
    _render_done_info = {}

    def _render_init(self, render_info):
        if not self.render_flag and not self.traj_flag:
            print("Without pic and data saved")
            return
        else:
            print(
                f"Into the data verbose mode. render: {self.render_flag}, traj: {self.traj_flag}")

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
                "render_type": render_info["type"],  # pic type
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
                "render_type": render_info["type"],  # pic type
                "render_config": render_info,
            })
            # self._render_cfg.save()

        if self.render_flag:
            f = plt.figure(figsize=(16, 9))
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
            "_debug_done_errlat": self._debug_done_errlat,
            # "_debug_done_errlon"  : self._debug_done_errlon ,
            "_debug_done_errhead": self._debug_done_errhead,
            "_debug_done_postype": self._debug_done_postype,
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

        append_to_pickle_incremental(os.path.join(
            self._render_cfg._debug_path_qxdata, "trajs.pkl"), obj)
        pass

    def _render_update_info(self, mf_info, *, add_info={}):
        self._render_info = {}
        for tag in self._render_tags:
            if tag in mf_info.keys() and mf_info[tag] is not None:
                self._render_info[tag] = mf_info[tag]
            # self._render_info.update(add_info)

    def _render_sur_byobs(self, neighbor_info=None, color='black', *, save_func=None, show_done=True, show_debug=False, **kwargs):
        if self.traj_flag:
            self._render_save_traj()
        if not self.render_flag:
            return

        original_nei = self._render_surcars

        f, ax = plt.gcf(), plt.gca()
        ego_x, ego_y = self._state[0], self._state[1]
        phi = self._state[4]
        dx, dy = self._render_cfg.arrow_len * \
            np.cos(phi), self._render_cfg.arrow_len*np.sin(phi)
        arrow = plt.arrow(ego_x, ego_y, dx, dy, head_width=0.5)
        plt.xlim(ego_x-self._render_cfg.draw_bound,
                 ego_x+self._render_cfg.draw_bound)
        plt.ylim(ego_y-self._render_cfg.draw_bound,
                 ego_y+self._render_cfg.draw_bound)
        dot = plt.scatter(ego_x, ego_y, color='red', s=10)
        self._render_ego_shadows.append((dot, arrow))

        ref_x, ref_y = self._ref_points[:, 0], self._ref_points[:, 1]
        ref_lines = plt.plot(ref_x, ref_y, ls="dotted",
                             color="red", linewidth=8)

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
            text_strs += [f"{key}:{val};" for key,
                          val in self._render_done_info.items()]
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
                color=color  #
            )
            text_objs.append(text_obj)

        def draw_car(center_x, center_y, length, width, phi, facecolor="lightblue", id=None):
            car = patches.Rectangle(
                # Bottom-left corner of the rectangle
                (center_x - length / 2, center_y - width / 2),
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
                car_rectangles.append(
                    draw_car(center_x, center_y, length, width, phi, obj_id))
                plt.gca().add_patch(car_rectangles[-1][0])

        ego_car_t = draw_car(float(self._ego[0]), float(self._ego[1]), float(
            self._ego[4]), float(self._ego[5]), self._state[4], "pink", "ego")
        plt.gca().add_patch(ego_car_t[0])
        car_rectangles.append(ego_car_t)

        # saving
        if save_func is None:
            self._render_count += 1
            f.savefig(os.path.join(f"{self._render_cfg._debug_path_qxdata}", str(
                self._render_count) + self._render_cfg.render_type), dpi=self._render_cfg["dpi"])
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
