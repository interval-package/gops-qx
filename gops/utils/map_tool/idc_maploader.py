import random
import json
import time
import numpy as np
from matplotlib import cm as cmx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import gops.utils.map_tool.proto.hdmap_pb2 as hdmap_pb2
import google.protobuf.text_format as text_format
from google.protobuf import json_format
from dataclasses import dataclass
from typing import List
from gops.utils.map_tool.utils import *

class MapBase:
    def __init__(self) -> None:
        self.map_pb = hdmap_pb2.HdMap()

        self.lane2idx = dict()
        self.link2idx = dict()
        self.segment2idx = dict()
        self.junction2idx = dict()
        self.connection2idx = dict()

        self.lane_id_list = []
        self.link_id_list = []
        self.segment_id_list = []
        self.junction_id_list = []
        self.connection_id_list = []

    def load(self, map_file_name):
        with open(map_file_name, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        json_str = json.dumps(json_obj, indent=4)
        json_to_pb = json_format.Parse(json_str, self.map_pb)
        for i, lane in enumerate(self.map_pb.lanes):
            self.lane2idx[lane.lane_id] = i
            self.lane_id_list.append(lane.lane_id)

        for i, link in enumerate(self.map_pb.links):
            self.link2idx[link.link_id] = i
            self.link_id_list.append(link.link_id)

        for i, segment in enumerate(self.map_pb.segments):
            self.segment2idx[segment.segment_id] = i
            self.segment_id_list.append(segment.segment_id)

        for i, junction in enumerate(self.map_pb.junctions):
            self.junction2idx[junction.junction_id] = i
            self.junction_id_list.append(junction.junction_id)
        
        for i, connection in enumerate(self.map_pb.connections):
            self.connection2idx[connection.connection_id] = i
            self.connection_id_list.append(connection.connection_id)

    def get_lane_by_id(self, id):
        assert id in self.lane_id_list, "lane id `{}` not in map".format(id)
        idx = self.lane2idx[id]
        return self.map_pb.lanes[idx]

    def get_link_by_id(self, id):
        try:
            assert id in self.link_id_list
        except Exception as e:
            print("warning: make sure link `{}` is useless".format(id))
        idx = self.link2idx[id]
        return self.map_pb.links[idx]

    def get_segment_by_id(self, id):
        assert id in self.segment_id_list
        idx = self.segment2idx[id]
        return self.map_pb.segments[idx]

    def get_junction_by_id(self, id):
        assert id in self.junction_id_list
        idx = self.junction2idx[id]
        return self.map_pb.junctions[idx]
    
    def get_connection_by_id(self, id):
        assert id in self.connection_id_list
        idx = self.connection2idx[id]
        return self.map_pb.connections[idx]
    
    def get_link_id_by_lane_id(self, lane_id):
        lane = self.get_lane_by_id(lane_id)
        return lane.link_id
    
    def get_segment_id_by_link_id(self, link_id):
        link = self.get_link_by_id(link_id)
        return link.segment_id
    
    def get_segment_id_by_lane_id(self, lane_id):
        link_id = self.get_link_id_by_lane_id(lane_id)
        return self.get_segment_id_by_link_id(link_id)
    
    def get_junction_id_by_connection_id(self, connection_id):
        connection = self.get_connection_by_id(connection_id)
        return connection.junction_id
    
    # 获取当前link的所有lane_id，需要排除无效的lane
    def get_lane_id_list_by_link_id(self, link_id):
        link = self.get_link_by_id(link_id)
        lane_id_list = []
        for lane_id in link.ordered_lane_ids:
            lane = self.get_lane_by_id(lane_id)
            if len(lane.center_line) > 0:
                lane_id_list.append(lane_id)
        return lane_id_list
    
    # 获取当前link的下游link_list
    def get_next_link_list_by_link_id(self, link_id):
        # 获取当前link的所有lane_id
        link = self.get_link_by_id(link_id)
        lane_id_list = link.ordered_lane_ids

        next_link_set = set()
        
        for lane_id in lane_id_list:
            lane = self.get_lane_by_id(lane_id)
            if not lane:
                continue
            
            next_link_id_list = lane.downstream_link_ids
            for next_link_id in next_link_id_list:
                next_link_set.add(next_link_id)  # 添加到集合中

        return list(next_link_set)  # 将集合转换为列表并返回
    
    # 获取两个link之间的connection_id，如果没有则返回[]
    def get_connection_id_list_by_link_id(self, upstream_link_id, downstream_link_id):
        # 遍历connection，找到upstream_link_id和downstream_link_id对应的connection
        connection_id_list = []
        for connection_id in self.connection_id_list:
            connection = self.get_connection_by_id(connection_id)
            if connection.upstream_link_id == upstream_link_id and connection.downstream_link_id == downstream_link_id:
                connection_id_list.append(connection_id)
        return connection_id_list
    
    # 判断一个lane_id是connection还是lane
    def tell_lane_id_is_connection(self, lane_id):
        if lane_id in self.connection_id_list:
            return True
        elif lane_id in self.lane_id_list:
            return False
        else:
            raise ValueError("lane_id `{}` not in map".format(lane_id))
    
    def get_ref_path_by_id(self, lane_id):
        if self.tell_lane_id_is_connection(lane_id):
            return self.get_ref_path_by_connection_id(lane_id)
        else:
            return self.get_ref_path_by_lane_id(lane_id)
    
    # 根据lane_id生成参考路径，注意这是没有离散化过的
    def get_ref_path_by_lane_id(self, lane_id):
        lane = self.get_lane_by_id(lane_id)
        ref_path_x = [point.x for point in lane.center_line]
        ref_path_y = [point.y for point in lane.center_line]
        ref_path_phi = xy2phi(ref_path_x, ref_path_y)
        ref_path_speed = np.zeros_like(ref_path_x) + 10.0
        ref_path = RefPath(
            path=np.array([ref_path_x, ref_path_y, ref_path_phi, ref_path_speed]),
            lane_id_list=[lane_id],
            link_id_list=[lane.link_id],
            road_id_list=[self.get_segment_id_by_link_id(lane.link_id)]
        )
        return ref_path
    
    # 根据connnection_id生成参考路径，注意这是没有离散化过的
    def get_ref_path_by_connection_id(self, connection_id):
        connection = self.get_connection_by_id(connection_id)
        ref_path_x = [point.x for point in connection.path]
        ref_path_y = [point.y for point in connection.path]
        ref_path_phi = xy2phi(ref_path_x, ref_path_y)
        ref_path_speed = np.zeros_like(ref_path_x) + 10.0 # TODO: 速度规划，初步打算根据曲率来规划速度
        ref_path = RefPath(
            path=np.array([ref_path_x, ref_path_y, ref_path_phi, ref_path_speed]),
            lane_id_list=[connection.connection_id],
            link_id_list=["connection"], # connection没有link_id，处理时要注意
            road_id_list=[self.get_junction_id_by_connection_id(connection.connection_id)]
        )
        return ref_path
    
    # 判断两条lane是否相连
    def judge_lane_connected(self, upstream_lane_id, downstream_lane_id):
        # 判断两条lane是否相连，如果相连返回True，否则返回False
        up_x_end = 0.0
        up_y_end = 0.0
        down_x_start = 0.0
        down_y_start = 0.0
        if self.tell_lane_id_is_connection(upstream_lane_id):
            connection = self.get_connection_by_id(upstream_lane_id)
            if len(connection.path) == 0:
                return False
            up_x_end = connection.path[-1].x
            up_y_end = connection.path[-1].y
        else:
            lane = self.get_lane_by_id(upstream_lane_id)
            if len(lane.center_line) == 0:
                return False
            up_x_end = lane.center_line[-1].x
            up_y_end = lane.center_line[-1].y
        
        if self.tell_lane_id_is_connection(downstream_lane_id):
            connection = self.get_connection_by_id(downstream_lane_id)
            if len(connection.path) == 0:
                return False
            down_x_start = connection.path[0].x
            down_y_start = connection.path[0].y
        else:
            lane = self.get_lane_by_id(downstream_lane_id)
            if len(lane.center_line) == 0:
                return False
            down_x_start = lane.center_line[0].x
            down_y_start = lane.center_line[0].y
        
        distance = np.sqrt((up_x_end - down_x_start)**2 + (up_y_end - down_y_start)**2)

        if distance < 0.5:
            return True
        else:
            return False

    
    # 获取当前lane的行驶方向
    def get_lane_direction(self, lane_id: str) -> str:
        lane_path = self.get_ref_path_by_id(lane_id)
        if not lane_path:
            return "unknown"

        lane_path_phi = lane_path.path[2, :]

        # 计算航向角变化，考虑越过 ±π 的情况
        delta_phi = np.diff(lane_path_phi)
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # 将角度差归一化到 [-π, π]

        # 计算累计的航向角变化
        cumulative_phi_change = np.sum(delta_phi)
        
        # 判断累计变化量
        if cumulative_phi_change > np.radians(30):
            return "left"
        elif cumulative_phi_change < np.radians(-30):
            return "right"
        elif cumulative_phi_change > np.radians(120):
            return "uturn"
        else:
            return "straight"
    
    # 获取当前link的行驶方向
    def get_link_direction(self, link_id):
        # 获取一条lane的center_line，根据center_line的弯曲程度判断行驶方向
        lane_id_list = self.get_link_by_id(link_id).ordered_lane_ids
        one_valid_lane_id = None
        for lane_id in lane_id_list:
            lane = self.get_lane_by_id(lane_id)
            if (len(lane.center_line) > 0):
                one_valid_lane_id = lane_id
                break
        
        if one_valid_lane_id is None:
            raise ValueError("link_id `{}` has no path".format(link_id))

        return self.get_lane_direction(one_valid_lane_id)
    
    # 根据千行给出的link_route，判断这辆车的行驶工况
    # 返回值：left, right, straight, uturn, other
    # other包括了无效的link_route，以及多种行驶工况的情况
    # TODO: 需要排除uturn
    def get_route_direction(self, link_route: List[str]):
        # route中包含若干link，[link1, link2, link3, ...]，可能只有一个link
        # 如果两个link不直接相连，并且之间没有connection，那么返回other
        direction_list = []
        assert len(link_route) > 0, "link_route is empty"
        
        for i in range(len(link_route) - 1):
            #获取当前link的所有下游link
            next_link_list = self.get_next_link_list_by_link_id(link_route[i])
            if link_route[i+1] in next_link_list:
                direction_list.append(self.get_link_direction(link_route[i]))
            else:
                connection_id_list = self.get_connection_id_list_by_link_id(link_route[i], link_route[i+1])
                if len(connection_id_list) == 0:
                    return "other"
                else:
                    direction_list.append(self.get_lane_direction(connection_id_list[0]))
        
        # 判断行驶方向，如果都是直行，则返回straight，如果是直行加左转或者直行加右转，则返回左转或者右转，如果左右转都有，则返回other
        if "left" in direction_list and "right" in direction_list:
            return "other"
        elif "left" in direction_list:
            return "left"
        elif "right" in direction_list:
            return "right"
        else:
            return "straight"

if __name__ == "__main__":
    map = MapBase()
    map.load("./map/crossroads_map_refined.json")

    # 测试get_ref_path_by_lane_id和get_ref_path_by_connection_id
    ref_path_1 = map.get_ref_path_by_lane_id("sg5_lk0_la3")
    print(ref_path_1)
    ref_path_2 = map.get_ref_path_by_connection_id("sg2_lk0_la1_sg1_lk0_la3")
    print(ref_path_2)
    
    plt.plot(ref_path_1.path[0, :], ref_path_1.path[1, :], 'r')
    plt.plot(ref_path_2.path[0, :], ref_path_2.path[1, :], 'b')
    plt.axis("equal")
    plt.savefig("ref_path.png")

    # 测试get_lane_id_list_by_link_id
    print("=====================================")
    lane_id_list = map.get_lane_id_list_by_link_id("sg8_lk0")
    print("lane_id_list of sg8_lk0: ", lane_id_list)

    # 测试get_next_link_list_by_link_id
    print("=====================================")
    next_link_list = map.get_next_link_list_by_link_id("sg5_lk0")
    print("next_link_list of sg5_lk0: ", next_link_list)

    # 测试get_connection_id_list_by_link_id
    print("=====================================")
    connection_id_list = map.get_connection_id_list_by_link_id("sg8_lk0", "sg1_lk0")
    print("connection_id_list of sg8_lk0 and sg5_lk0: ", connection_id_list)

    # 测试get_lane_direction
    print("=====================================")
    lane_direction_1 = map.get_lane_direction("sg8_lk0_la1")
    lane_direction_2 = map.get_lane_direction("sg8_lk0_la3_sg1_lk0_la1")
    lane_direction_3 = map.get_lane_direction("sg5_lk0_la1")
    print("lane_direction of sg8_lk0_la1: ", lane_direction_1)
    print("lane_direction of sg8_lk0_la3_sg1_lk0_la1: ", lane_direction_2)
    print("lane_direction of sg5_lk0_la1: ", lane_direction_3)

    # 测试get_link_direction
    print("=====================================")
    link_direction_1 = map.get_link_direction("sg8_lk0")
    print("link_direction of sg8_lk0: ", link_direction_1)

    # 测试get_route_direction
    print("=====================================")
    route_1 = ["sg7_lk0"]
    route_2 = ["sg8_lk0", "sg1_lk0"]
    route_3 = ["sg4_lk0", "sg2_lk0"]
    route_direction_1 = map.get_route_direction(route_1)
    route_direction_2 = map.get_route_direction(route_2)
    route_direction_3 = map.get_route_direction(route_3)
    print("route_direction of route_1: ", route_direction_1)
    print("route_direction of route_2: ", route_direction_2)
    print("route_direction of route_3: ", route_direction_3)

    # 测试judge_lane_connected
    print("=====================================")
    connected_1 = map.judge_lane_connected("sg8_lk0_la3", "sg8_lk0_la3_sg1_lk0_la2")
    connected_2 = map.judge_lane_connected("sg8_lk0_la3", "sg1_lk0_la2")
    connected_3 = map.judge_lane_connected("sg8_lk0_la3_sg1_lk0_la2", "sg1_lk0_la2")
    print("sg8_lk0_la3 and sg8_lk0_la3_sg1_lk0_la2 connected: ", connected_1)
    print("sg8_lk0_la3 and sg1_lk0_la2 connected: ", connected_2)
    print("sg8_lk0_la3_sg1_lk0_la1 and sg1_lk0_la2 connected: ", connected_3)
