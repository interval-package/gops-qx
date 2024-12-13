import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

from dataclasses import dataclass
from gops.utils.map_tool.idc_maploader import MapBase, RefPath
from typing import List
from gops.utils.map_tool.utils import *

class IDCStaticPlanner:
    def __init__(self, **kwargs) -> None:
        self.map_path = kwargs.get('map_path')
        self.link_route = kwargs.get('link_route', [])
        self.map = MapBase()

        self.load_map(self.map_path)
        self.global_static_paths = self.generate_global_static_paths()
    
    def load_map(self, map_path: str):
        self.map = MapBase()
        self.map.load(map_path)
    
    def set_route(self, link_route: List[str]):
        self.link_route = link_route
        self.global_static_paths = self.generate_global_static_paths()
    
    # 根据link_route生成全局静态路径
    # 首先列出所有的lane_id，然后根据lane的连通关系生成路径
    def generate_global_static_paths(self) -> List[RefPath]:
        static_path = []
        lane_route_list = []
        if len(self.link_route) > 0:
            lane_id_list_list = [] # 二维列表，每个元素是一个link或junction的lane_id_list
            lane_id_list_list.append(self.map.get_lane_id_list_by_link_id(self.link_route[0]))
            for i in range(len(self.link_route) - 1):
                # 判断当前link是否直通下一个link，如果是则直接添加下一个link的lane_id_list
                # 如果不是，则判断两者之间有无connection，如果有则添加connection的lane_id_list
                # 如果还不是，则说明不连通，报错
                next_link_list = self.map.get_next_link_list_by_link_id(self.link_route[i])
                if self.link_route[i + 1] in next_link_list:
                    lane_id_list_list.append(self.map.get_lane_id_list_by_link_id(self.link_route[i + 1]))
                else:
                    connection_list = self.map.get_connection_id_list_by_link_id(
                        upstream_link_id=self.link_route[i], 
                        downstream_link_id=self.link_route[i + 1]
                        )
                    if len(connection_list) > 0:
                        lane_id_list_list.append(connection_list)
                        lane_id_list_list.append(self.map.get_lane_id_list_by_link_id(self.link_route[i + 1]))
                    else:
                        raise Exception(f"Link {self.link_route[i]} and Link {self.link_route[i + 1]} are not connected")
            # print("lane_id_list_list:")
            # for sublist in lane_id_list_list:
            #     print(sublist)
            
            # 根据连通性，获取所有从lane_id_list_list[0]到lane_id_list_list[-1]的连通路径
            # 连通性根据self.map.judge_lane_connected(lane_id_1, lane_id_2)来判断
            # 递归回溯法
            def find_all_paths(lane_id_list_list):
                def backtrack(current_path, current_index):
                    if current_index == len(lane_id_list_list) - 1:
                        lane_route_list.append(current_path.copy())
                        return
                    
                    for next_lane in lane_id_list_list[current_index + 1]:
                        if self.map.judge_lane_connected(current_path[-1], next_lane):
                            current_path.append(next_lane)
                            backtrack(current_path, current_index + 1)
                            current_path.pop()

                lane_route_list = []
                for start_lane in lane_id_list_list[0]:
                    backtrack([start_lane], 0)
                return lane_route_list

            lane_route_list = find_all_paths(lane_id_list_list)
            # print(":")
            # for lane_route in lane_route_list:
            #     print(lane_route)
            
            # 根据lane_route_list生成静态路径
            for lane_route in lane_route_list:
                static_path.append(self.generate_path_by_lane_id_list(lane_route))

        return static_path

    # 根据lane_id_list生成静态路径
    def generate_path_by_lane_id_list(self, lane_id_list: List[str]) -> RefPath:
        path_list = []
        for lane_id in lane_id_list:
            path_list.append(self.map.get_ref_path_by_id(lane_id))
        
        return concat_ref_paths(path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="./map/crossroads_map_refined.json")
    args = parser.parse_args()

    planner = IDCStaticPlanner(map_path=args.map_path)
    
    # 测试generate_path_by_lane_id_list
    print("========== Test generate_path_by_lane_id_list ==========")
    lane_id_list = ["sg8_lk0_la3", "sg8_lk0_la3_sg1_lk0_la2", "sg1_lk0_la2"]
    ref_path = planner.generate_path_by_lane_id_list(lane_id_list)
    print(ref_path)

    plt.plot(ref_path.path[0], ref_path.path[1])
    plt.axis("equal")
    plt.savefig("test_static_planner.png", dpi=800)

    # 测试generate_global_static_paths
    print("========== Test generate_global_static_paths ==========")
    route_1 = ["sg7_lk0"]
    route_2 = ["sg8_lk0", "sg1_lk0"]
    route_3 = ["sg4_lk0", "sg5_lk0"]
    planner.set_route(route_1)
    planner.set_route(route_2)
    planner.set_route(route_3)
    print("global_static_paths:")
    for ref_path in planner.global_static_paths:
        print(ref_path)

    # 绘制全局静态路径
    plt.figure()
    for ref_path in planner.global_static_paths:
        plt.plot(ref_path.path[0], ref_path.path[1])
    plt.axis("equal")

    # 测试pathcut
    ego_x = -200
    ego_y = 0
    ref_paths_cut = []
    for ref_path in planner.global_static_paths:
        ref_path_cut = path_cut(ref_path, ego_x, ego_y)
        ref_paths_cut.append(ref_path_cut)
        plt.plot(ref_path_cut.path[0], ref_path_cut.path[1], 'r')
    
    # 测试path_discrete_t
    ref_paths_discrete = []
    for ref_path_cut in ref_paths_cut:
        ref_path_discrete = path_discrete_t(ref_path_cut, 0.1)
        ref_paths_discrete.append(ref_path_discrete)
        plt.plot(ref_path_discrete.path[0], ref_path_discrete.path[1], 'b')

    plt.savefig("test_static_planner_global.png", dpi=300)    




