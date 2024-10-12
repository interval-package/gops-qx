import json
import math
import random

import google.protobuf.text_format as text_format
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from google.protobuf import json_format
from matplotlib import cm as cmx
from matplotlib import colors as mcolors
from matplotlib.path import Path as plt_Path

import proto.hdmap_attach_pb2 as hdmap_attach_pb2
import proto.hdmap_pb2 as hdmap_pb2
from lib.map_base import MapBase
from lib.rule import ADD_JC, DELETE_JC


class Map2(MapBase):
    def __init__(self, map_name) -> None:
        super().__init__()
        self._map_name = map_name
        self.pb_attach_map = hdmap_attach_pb2.HdMapAttach()
        self.l2j_list = []
        self.j2l_list = []
        self.j2link_list = []
        self.l2j_id = dict()

    def write_l2j_j2l(self):
        for segment in self.map_pb.segments:
            real_ordered_link_ids = self._rank_links(segment.ordered_link_ids)
            if (len(real_ordered_link_ids)==0):
                continue
            start_link_id = real_ordered_link_ids[0]
            start_link = self.get_link_by_id(start_link_id)
            end_link_id = real_ordered_link_ids[-1]
            end_link = self.get_link_by_id(end_link_id)

            start_junction_id = segment.start_junction_id
            end_junction_id = segment.end_junction_id
            self.j2link_list.append(start_link_id)

            for lane_id in start_link.ordered_lane_ids:
                self.j2l_list.append(lane_id)

            for lane_id in end_link.ordered_lane_ids:
                self.l2j_list.append(lane_id)
                self.l2j_id[lane_id] = end_junction_id

        for lane in self.map_pb.lanes:
            lane_id = lane.lane_id
            for link_id in lane.connect_link_ids:
                link = self.get_link_by_id(link_id)

                if link.type == "pre_turn_right":
                    self.l2j_list.append(lane_id)
                    for other_lane_id in link.ordered_lane_ids:
                        if other_lane_id in self.l2j_id.keys():
                            self.l2j_id[lane_id] = self.l2j_id[other_lane_id]

        self.l2j_list = list(set(self.l2j_list))
        self.j2l_list = list(set(self.j2l_list))

        # write to proto
        for t in self.l2j_list:
            self.pb_attach_map.lane2junction.append(t)
        for t in self.j2l_list:
            self.pb_attach_map.junction2lane.append(t)

    def _tell_turning(self, lane, next_lane):
        assert len(lane.center_line) >= 2
        assert len(next_lane.center_line) >= 2

        dx = lane.center_line[-2].x - lane.center_line[-1].x
        dy = lane.center_line[-2].y - lane.center_line[-1].y

        lane_phi = math.atan2(dy, dx)

        ndx = next_lane.center_line[0].x - next_lane.center_line[1].x
        ndy = next_lane.center_line[0].y - next_lane.center_line[1].y

        next_lane_phi = math.atan2(ndy, ndx)

        dphi = lane_phi - next_lane_phi

        dphi = dphi % (2 * 3.14)
        # print(lane.lane_id, next_lane.lane_id)
        # print(dphi)
        if dphi >= math.radians(45) and dphi <= math.radians(135):
            return "right"
        elif dphi >= math.radians(135) and dphi <= math.radians(225):
            return "uturn"
        elif dphi >= math.radians(225) and dphi <= math.radians(315):
            return "left"
        else:
            return "straight"

    def _get_straigh_ref(self, lane, next_lane):
        start_point = [lane.center_line[-1].x, lane.center_line[-1].y]
        end_point = [next_lane.center_line[0].x, next_lane.center_line[0].y]
        middle_point = [(start_point[0] + end_point[0]) * 0.5, (start_point[1] + end_point[1]) * 0.5]
        return [start_point, middle_point, end_point]

    def _get_control_points(self, lane, next_lane):
        start_point = [lane.center_line[-1].x, lane.center_line[-1].y]
        end_point = [next_lane.center_line[0].x, next_lane.center_line[0].y]
        first_point = start_point

        dx = lane.center_line[-2].x - lane.center_line[-1].x
        dy = lane.center_line[-2].y - lane.center_line[-1].y
        start_theta = math.atan2(dy, dx)

        ndx = next_lane.center_line[0].x - next_lane.center_line[1].x
        ndy = next_lane.center_line[0].y - next_lane.center_line[1].y
        end_theta = math.atan2(ndy, ndx)

        turn_dir = self._tell_turning(lane, next_lane)
        if (turn_dir == "right") and next_lane.lane_offset == 1:
            end_point[0] = end_point[0] - 10 * np.cos(end_theta)
            end_point[1] = end_point[1] - 10 * np.sin(end_theta)
        elif (turn_dir == "left") and next_lane.lane_offset == 1:
            end_point[0] = end_point[0] + 0 * np.cos(end_theta)
            end_point[1] = end_point[1] + 0 * np.sin(end_theta)

        # Reduced coefficients for closer control points
        coeff = 0.6  # Adjust this coefficient to change the proximity of control points

        A = [
            [coeff + (1-coeff) * np.sin(start_theta) * np.sin(start_theta), -0.5*(1-coeff) * np.sin(2 * start_theta)],
            [-0.5*(1-coeff) * np.sin(2 * start_theta), coeff + (1-coeff) * np.cos(start_theta) * np.cos(start_theta)],
        ]
        B = [
            [(1-coeff) * np.cos(start_theta) * np.cos(start_theta), 0.5*(1-coeff) * np.sin(2 * start_theta)],
            [0.5*(1-coeff) * np.sin(2 * start_theta), (1-coeff) * np.sin(start_theta) * np.sin(start_theta)],
        ]

        second_point = list((np.dot(A, start_point) + np.dot(B, end_point)))

        C = [
            [(1-coeff) * np.cos(end_theta) * np.cos(end_theta), 0.5*(1-coeff) * np.sin(2 * end_theta)],
            [0.5*(1-coeff) * np.sin(2 * end_theta), (1-coeff) * np.sin(end_theta) * np.sin(end_theta)],
        ]
        D = [
            [coeff + (1-coeff) * np.sin(end_theta) * np.sin(end_theta), -0.5*(1-coeff) * np.sin(2 * end_theta)],
            [-0.5*(1-coeff) * np.sin(2 * end_theta), coeff + (1-coeff) * np.cos(end_theta) * np.cos(end_theta)],
        ]

        third_point = list(np.dot(C, start_point) + np.dot(D, end_point))

        fourth_point = end_point
        return first_point, second_point, third_point, fourth_point

    def _rank_links(self, list_of_links):
        return list_of_links
        ranked_links = []
        for link_id in list_of_links:
            link = self.get_link_by_id(link_id)
            has_next = False
            for lane_id in link.ordered_lane_ids:
                lane = self.get_lane_by_id(lane_id)
                for next_link_id in lane.connect_link_ids:
                    if next_link_id in list_of_links:
                        has_next = True
            if not has_next:
                ranked_links.append(link_id)
                break

        while True:
            for link_id in list_of_links:
                print(link_id)
                link = self.get_link_by_id(link_id)
                has_next = False
                for lane_id in link.ordered_lane_ids:
                    lane = self.get_lane_by_id(lane_id)
                    for next_link_id in lane.connect_link_ids:
                        if next_link_id == ranked_links[0]:
                            ranked_links = [link_id] + ranked_links
            if len(ranked_links) == len(list_of_links):
                break

        return ranked_links

    def _tell_beizer_in_which_junction(self, ref):
        j_id = None
        for j in self.map_pb.junctions:
            shape = []
            for p in j.shape:
                shape.append([p.x, p.y])
            if self._tell_ref_in_junction(ref, shape):
                j_id = j.junction_id
                break
        assert j_id is not None
        return j_id

    def _tell_ref_in_junction(self, ref, junction_shape):
        path = plt_Path(junction_shape)
        idx = len(ref) // 2
        return path.contains_point(ref[idx])

    @staticmethod
    def _half_circle(start, start_phi, end, end_phi):
        extension_len = 2.0
        ell_param = 10.0
        point_per_meter = 30
        dm = 1.0 / point_per_meter
        sx, sy = start
        ex, ey = end
        shape = []
        start_shape = []
        # add start extantion
        for i in range(int(extension_len * point_per_meter)):
            start_shape.append([sx + i * np.cos(start_phi) * dm, sy + i * np.sin(start_phi) * dm])
        end_shape = []
        # add end extantion
        for i in range(int(extension_len * point_per_meter)):
            end_shape.append([ex + i * np.cos(end_phi) * dm, ey + i * np.sin(end_phi) * dm])
        end_shape.reverse()

        sx, sy = start_shape[-1]
        ex, ey = end_shape[0]
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        r = np.sqrt((sx - ex) ** 2 + (sy - ey) ** 2) / 2
        vecs = sx - mx, sy - my

        base_angle = np.arctan2(vecs[1], vecs[0])
        theta_ty = base_angle + 3.14159 * 0.5

        for i in range(15):
            shape.append(
                [mx + r * np.cos(base_angle + 3.14159 * i / 30), my + r * np.sin(base_angle + 3.14159 * i / 30)]
            )

        # add Ellipse
        base_angle = base_angle + 3.14159 * 0.5
        ell_start_point = shape[-1]

        ell_end_point = (
            end[0] - (ell_param - extension_len) * np.cos(end_phi),
            end[1] - (ell_param - extension_len) * np.sin(end_phi),
        )

        def foot_point_of_point_to_segment(x0, y0, x1, y1, x2, y2):
            # 如果两点相同，则输出一个点的坐标为垂足
            if x1 == x2 and y1 == y2:
                return x1, y1, True
            k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            xf = k * (x2 - x1) + x1
            yf = k * (y2 - y1) + y1
            flag = True
            if k < 0 or k > 1:
                flag = False
            return xf, yf, flag

        x_tmp, y_tmp, flag = foot_point_of_point_to_segment(
            ell_end_point[0], ell_end_point[1], ell_start_point[0], ell_start_point[1], mx, my
        )
        ell_mx = x_tmp
        ell_my = y_tmp
        ell_a = np.sqrt((x_tmp - ell_start_point[0]) ** 2 + (y_tmp - ell_start_point[1]) ** 2)
        ell_b = np.sqrt((x_tmp - ell_end_point[0]) ** 2 + (y_tmp - ell_end_point[1]) ** 2)

        ell_c = np.sqrt(ell_a**2 - ell_b**2)
        vecs = ell_start_point[0] - ell_mx, ell_start_point[1] - ell_my
        base_angle2 = np.arctan2(vecs[1], vecs[0])

        ell_origin_x = ell_mx - ell_c * np.cos(base_angle2)
        ell_origin_y = ell_my - ell_c * np.sin(base_angle2)

        max_amgle = np.arctan2(ell_b, ell_c)

        for i in range(30):
            ty_r = ell_b * ell_b / (ell_a - ell_c * np.cos(max_amgle * i / 30))
            shape.append(
                [
                    ell_origin_x + ty_r * np.cos(base_angle2 + max_amgle * i / 30),
                    ell_origin_y + ty_r * np.sin(base_angle2 + max_amgle * i / 30),
                ]
            )

        whole_shape = start_shape + shape
        return whole_shape

    @staticmethod
    def _beizer3(first_point, second_point, third_point, fourth_point):
        sub_arc_len = 1.0
        estimated_arc_len = 1.5 * math.sqrt(
            (first_point[0] - fourth_point[0]) * (first_point[0] - fourth_point[0])
            + (first_point[1] - fourth_point[1]) * (first_point[1] - fourth_point[1])
        )

        num = estimated_arc_len / sub_arc_len
        t = 0.0
        dt = 1.0 / num

        ref = []
        for i in range(int(num)):
            x1 = (
                (1 - t) * (1 - t) * (1 - t) * first_point[0]
                + 3 * t * (1 - t) * (1 - t) * second_point[0]
                + 3 * t * t * (1 - t) * third_point[0]
                + t * t * t * fourth_point[0]
            )
            y1 = (
                (1 - t) * (1 - t) * (1 - t) * first_point[1]
                + 3 * t * (1 - t) * (1 - t) * second_point[1]
                + 3 * t * t * (1 - t) * third_point[1]
                + t * t * t * fourth_point[1]
            )
            ref.append([x1, y1])
            t += dt

        ref.append(fourth_point)

        return ref

    def _get_curve_ref(self, lane, next_lane):
        control_points = self._get_control_points(lane, next_lane)
        ref = self._beizer3(*control_points)
        return ref

    def _search_all_possiable_roads(self, lane):
        link = self.get_link_by_id(lane.link_id)
        if link.type != "pre_turn_right":
            if len(lane.center_line) >= 2:
                return [[lane.lane_id]]
            else:
                [["failed"]]
        else:
            res = []
            for next_link_id in lane.connect_link_ids:
                next_link = self.get_link_by_id(next_link_id)
                for next_lane_id in next_link.ordered_lane_ids:
                    next_lane = self.get_lane_by_id(next_lane_id)
                    if len(next_lane.center_line) >= 2 and next_lane.type != "unknow":
                        c_dist = math.sqrt(
                            math.pow(next_lane.center_line[0].x - lane.center_line[-1].x, 2)
                            + math.pow(next_lane.center_line[0].y - lane.center_line[-1].y, 2)
                        )
                        if c_dist < 0.5:
                            for tmp in self._search_all_possiable_roads(next_lane):
                                res.append(tmp)
            if len(res) == 0:
                return [["failed"]]
            else:
                opt = []
                for sub0 in res:
                    for sub in res:
                        tmp = [lane.lane_id] + sub0
                        opt.append(tmp)

                return opt

    def _get_lane_offset(self, lane, link):
        current_link_lane_list = link.ordered_lane_ids
        current_link_offsets = []
        lanes_list = []
        for lllid in current_link_lane_list:
            lll = self.get_lane_by_id(lllid)
            if lll.type == "driving":
                current_link_offsets.append(lll.lane_offset)
                lanes_list.append(lll)
        current_lane_offset = lane.lane_offset

        return current_lane_offset, current_link_offsets, lanes_list

    def _get_index(self, offset, offsets_with_index):
        for i, offset_tmp in offsets_with_index:
            if offset_tmp == offset:
                return i
        return 1

    def _is_two_lane_start_near(self, lane, last_lane):
        x = lane.center_line[0].x
        y = lane.center_line[0].y

        lx = last_lane.center_line[0].x
        ly = last_lane.center_line[0].y
        if math.sqrt((x - lx) * (x - lx) + (y - ly) * (y - ly)) < 0.3:
            return True
        else:
            return False

    def _process_index(self, link, offsets_with_index, lanes_list):
        idx_ptr = offsets_with_index[0][0]
        new_offsets_with_index = []
        new_offsets_with_index.append(offsets_with_index[0])

        for i, oi in enumerate(offsets_with_index):
            if i == 0:
                continue
            index, offset = oi

            last_lane = lanes_list[i - 1]
            lane = lanes_list[i]
            if self._is_two_lane_start_near(lane, last_lane):
                new_offsets_with_index.append((idx_ptr, offset))
            else:
                idx_ptr += 1
                new_offsets_with_index.append((idx_ptr, offset))
        return new_offsets_with_index

    def tell_jc_exist(self, lane_id, next_lane_id):
        # for connection in self.map_pb.connections:
        #     if connection.upstream_lane_id == lane_id and connection.downstream_lane_id == next_lane_id:
        #         return True

        # return False
        if self._map_name in ADD_JC.keys():
            for tmp in ADD_JC[self._map_name]:
                if tmp["from"] == lane_id and tmp["to"] == next_lane_id:
                    return True
        if self._map_name in DELETE_JC.keys():
            for tmp in DELETE_JC[self._map_name]:
                if tmp["from"] == lane_id and tmp["to"] == next_lane_id:
                    return False
        exist = True
        lane = self.get_lane_by_id(lane_id)
        link = self.get_link_by_id(lane.link_id)
        current_lane_offset, current_link_offsets, current_lanes_list = self._get_lane_offset(lane, link)
        current_link_offsets_with_index = list(enumerate(current_link_offsets))
        current_driving_lane_num = len(current_link_offsets)

        current_lane_offset_index = self._get_index(current_lane_offset, current_link_offsets_with_index)

        next_lane = self.get_lane_by_id(next_lane_id)
        next_link = self.get_link_by_id(next_lane.link_id)
        next_lane_offset, next_link_offsets, next_lanes_list = self._get_lane_offset(next_lane, next_link)
        next_link_offsets_with_index = list(enumerate(next_link_offsets))
        if len(next_link_offsets_with_index) == 0:
            return False
        next_link_offsets_with_index = self._process_index(next_link, next_link_offsets_with_index, next_lanes_list)
        next_lane_offset_index = self._get_index(next_lane_offset, next_link_offsets_with_index)
        direction = self._tell_turning(lane, next_lane)
        if direction == "uturn":
            return True

        # if current_lane_offset_index > len(next_link_offsets) - 1:
        #     delta_index = len(current_link_offsets) - len(next_link_offsets)
        #     if next_lane_offset_index != next_link_offsets_with_index[-1][0]:
        #         continue
        # if len(current_link_offsets) > len(next_link_offsets) - 1:
        #     delta_index = len(current_link_offsets) - len(next_link_offsets)
        #     if current_lane_offset_index - delta_index != next_lane_offset_index:
        #         exist = False
        if current_lane_offset == next_lane_offset:
            return True
        if len(current_link_offsets) > len(next_link_offsets) - 1:
            delta_index = len(current_link_offsets) - len(next_link_offsets)
            if current_lane_offset_index - delta_index != next_lane_offset_index:
                exist = False
            # if next_lane_offset_index != next_link_offsets_with_index[-delta_index][0]:
            #     continue
        else:
            if current_lane_offset_index != next_lane_offset_index:
                exist = False
        return exist

    def write_junction(self):
        # 没有考虑掉头
        junctions_connection_list = []
        for lane_id in self.l2j_list:
            lane = self.get_lane_by_id(lane_id)
            link = self.get_link_by_id(lane.link_id)
            junction_id = self.l2j_id[lane_id]
            if (lane_id == "457#_304523246_0_-1"):
                print("debug")

            for next_link_id in lane.connect_link_ids:
                next_link = self.get_link_by_id(next_link_id)
                if (lane_id == "457#_304523246_0_-1"):
                    print("lane_id: ", lane_id, "next_link_id: ", next_link_id)
                for next_lane_id in next_link.ordered_lane_ids:
                    if (lane_id == "457#_304523246_0_-1"):
                        print("lane_id: ", lane_id, "next_lane_id: ", next_lane_id)
                    next_lane = self.get_lane_by_id(next_lane_id)
                    if (len(lane.center_line) < 2 or len(next_lane.center_line) < 2) or next_lane.type == "unknow":
                        if (lane_id == "457#_304523246_0_-1"):
                            print("continue")
                        continue

                    if (lane_id == "457#_304523246_0_-1"):
                        print("tell_jc_exist: ", self.tell_jc_exist(lane_id, next_lane_id))
                    if not self.tell_jc_exist(lane_id, next_lane_id):
                        continue

                    c_dist = math.sqrt(
                        math.pow(lane.center_line[-1].x - next_lane.center_line[0].x, 2)
                        + math.pow(lane.center_line[-1].y - next_lane.center_line[0].y, 2)
                    )
                    if (next_lane_id in self.j2l_list) and c_dist > 0.5:
                        # beizer

                        if len(lane.center_line) >= 2 and len(next_lane.center_line) >= 2:
                            next_lane = self.get_lane_by_id(next_lane_id)
                            direction = self._tell_turning(lane, next_lane)
                            tmp_dict = dict()

                            tmp_dict["junction_id"] = junction_id
                            tmp_dict["from_lane"] = lane_id
                            tmp_dict["from_link"] = link.link_id
                            tmp_dict["from_segment"] = link.segment_id
                            tmp_dict["to_lane"] = next_lane_id
                            tmp_dict["to_link"] = next_link.link_id
                            tmp_dict["to_segment"] = next_link.segment_id
                            tmp_dict["connection_id"] = lane_id + "_" + next_lane_id
                            tmp_dict["direction"] = direction
                            if direction == "straight":
                                ref = self._get_straigh_ref(lane, next_lane)
                                tmp_dict["center_line"] = ref
                                tmp_dict["type"] = "beizer1"

                            elif direction in ["left", "right"]:
                                ref = self._get_curve_ref(lane, next_lane)
                                tmp_dict["center_line"] = ref
                                tmp_dict["type"] = "beizer3"
                            elif direction == "uturn":
                                # ref = self._get_straigh_ref(lane, next_lane) # todo
                                hc_start_point = [lane.center_line[-1].x, lane.center_line[-1].y]
                                hc_start_phi = math.atan2(
                                    lane.center_line[-1].y - lane.center_line[-2].y,
                                    lane.center_line[-1].x - lane.center_line[-2].x,
                                )
                                hc_end_point = [next_lane.center_line[0].x, next_lane.center_line[0].y]
                                hc_end_phi = math.atan2(
                                    next_lane.center_line[0].y - next_lane.center_line[1].y,
                                    next_lane.center_line[0].x - next_lane.center_line[1].x,
                                )
                                ref = self._half_circle(hc_start_point, hc_start_phi, hc_end_point, hc_end_phi)
                                tmp_dict["center_line"] = ref
                                tmp_dict["type"] = "half_circle"
                            else:
                                print(direction)
                                raise RuntimeError

                            # todo uturn
                            # try:
                            #     position_j_id = self._tell_beizer_in_which_junction(tmp_dict["center_line"])
                            # except Exception as e:
                            #     print(tmp_dict)
                            #     exit()
                            # tmp_dict["junction_id"] = position_j_id

                            junctions_connection_list.append(tmp_dict)
                    else:
                        # print(lane_id, next_lane_id)
                        if len(next_lane.center_line) >= 2 and next_lane.type != "unknow":
                            c_dist = math.sqrt(
                                math.pow(lane.center_line[-1].x - next_lane.center_line[0].x, 2)
                                + math.pow(lane.center_line[-1].y - next_lane.center_line[0].y, 2)
                            )
                            if c_dist <= 0.5:
                                res = self._search_all_possiable_roads(next_lane)
                                for sub_route in res:
                                    full_sub = [lane_id] + sub_route
                                if full_sub[-1] != "failed":
                                    tmp_dict = dict()
                                    tmp_dict["junction_id"] = self.l2j_id[lane_id]
                                    tmp_dict["from_lane"] = lane_id
                                    tmp_dict["from_link"] = link.link_id
                                    tmp_dict["from_segment"] = link.segment_id
                                    tmp_dict["to_lane"] = full_sub[-1]
                                    real_next_lane = self.get_lane_by_id(full_sub[-1])
                                    real_next_link_id = real_next_lane.link_id
                                    real_next_link = self.get_link_by_id(real_next_link_id)
                                    tmp_dict["to_link"] = real_next_lane.link_id
                                    tmp_dict["to_segment"] = real_next_link.segment_id
                                    tmp_dict["connection_id"] = lane_id + "_" + real_next_link_id
                                    tmp_dict["direction"] = "right"
                                    tmp_dict["type"] = "direct"
                                    tmp_dict["middle_lanes"] = full_sub[1:-1]
                                    junctions_connection_list.append(tmp_dict)

        # print([type(ele) for ele in junctions_connection_list])
        for ele in junctions_connection_list:
            obj = self.pb_attach_map.junction_connections.add()
            obj.junction_id = ele["junction_id"]
            obj.from_lane = ele["from_lane"]
            obj.from_link = ele["from_link"]
            obj.from_segment = ele["from_segment"]
            obj.to_lane = ele["to_lane"]
            obj.to_link = ele["to_link"]
            obj.to_segment = ele["to_segment"]
            obj.connection_id = ele["connection_id"]
            if ele["direction"] == "right":
                obj.direction = hdmap_attach_pb2.RIGHT
            elif ele["direction"] == "left":
                obj.direction = hdmap_attach_pb2.LEFT
            elif ele["direction"] == "straight":
                obj.direction = hdmap_attach_pb2.STRAIGHT
            elif ele["direction"] == "uturn":
                obj.direction = hdmap_attach_pb2.UTURN
            else:
                raise RuntimeError

            if ele["type"] == "direct":
                obj.type = hdmap_attach_pb2.DIRECT
            elif ele["type"] == "beizer1":
                obj.type = hdmap_attach_pb2.BEIZER1
            elif ele["type"] == "beizer3":
                obj.type = hdmap_attach_pb2.BEIZER3
            elif ele["type"] == "half_circle":
                obj.type = hdmap_attach_pb2.HALF_CIRCLE
            else:
                # print("ele[type]: ", ele["type"])
                raise RuntimeError
            if ele["type"] == "direct":
                for ttt in ele["middle_lanes"]:
                    obj.middle_lanes.append(ttt)
            else:
                for rx, ry in ele["center_line"]:
                    p = obj.center_line.add()
                    p.x = rx
                    p.y = ry

    def write_lanes(self):
        data = dict()  # keys: lane_id  --> dict: pre -> list suc -> list

        for lane in self.map_pb.lanes:
            lane_id = lane.lane_id
            data[lane_id] = dict()
            link_id = lane.link_id
            link = self.get_link_by_id(link_id)
            data[lane_id]["link_id"] = link_id
            data[lane_id]["segment_id"] = link.segment_id
            data[lane_id]["pre"] = []
            data[lane_id]["suc"] = []
            data[lane_id]["direction"] = []
            data[lane_id]["junction_connection"] = []
            for next_link_ids in lane.connect_link_ids:
                next_link = self.get_link_by_id(next_link_ids)
                for next_lane_id in next_link.ordered_lane_ids:
                    next_lane = self.get_lane_by_id(next_lane_id)
                    if (len(next_lane.center_line) >= 2) and next_lane != "unknow" and (len(lane.center_line) >= 2):
                        c_dist = math.sqrt(
                            math.pow(lane.center_line[-1].x - next_lane.center_line[0].x, 2)
                            + math.pow(lane.center_line[-1].y - next_lane.center_line[0].y, 2)
                        )
                        if c_dist <= 0.5:
                            data[lane_id]["suc"].append(next_lane_id)

        for lane_id in data.keys():
            for next_lane_id in data[lane_id]["suc"]:
                data[next_lane_id]["pre"].append(lane_id)

        for jc in self.pb_attach_map.junction_connections:
            from_lane_id = jc.from_lane
            data[from_lane_id]["junction_connection"].append(jc.connection_id)
            data[from_lane_id]["direction"].append(jc.direction)

        for lane_id in data.keys():
            if len(data[lane_id]["direction"]) > 0:
                succ_list = data[lane_id]["pre"]
                dirs = data[lane_id]["direction"]
                for i in range(3):
                    new_succ_list = []

                    for succ in succ_list:
                        if len(data[succ]["direction"]) == 0:
                            new_succ_list.extend(data[succ]["pre"])
                            data[succ]["direction"].extend(dirs)
                    succ_list = new_succ_list

        # //write to proto
        for lane_id in data.keys():
            obj = self.pb_attach_map.lanes.add()
            obj.lane_id = lane_id
            obj.link_id = data[lane_id]["link_id"]
            obj.segment_id = data[lane_id]["segment_id"]
            for d in list(set(data[lane_id]["direction"])):
                obj.directions.append(d)
            for d in data[lane_id]["pre"]:
                obj.pre_lanes.append(d)
            for d in data[lane_id]["suc"]:
                obj.suc_lanes.append(d)
            for d in data[lane_id]["junction_connection"]:
                obj.junction_connection.append(d)

    def run(self):
        self.write_l2j_j2l()
        self.write_junction()
        self.write_lanes()
        # print(self.pb_attach_map)

    def generate_half_circle_shape(self):
        try:
            lane = self.get_lane_by_id("sg24_lk1_la1")
            next_lane = self.get_lane_by_id("sg0_lk0_la1")
            hc_start_point = [lane.center_line[-1].x, lane.center_line[-1].y]
            hc_end_point = [next_lane.center_line[0].x, next_lane.center_line[0].y]
            hc_start_phi = math.atan2(
                lane.center_line[-1].y - lane.center_line[-2].y, lane.center_line[-1].x - lane.center_line[-2].x
            )
            hc_end_phi = math.atan2(
                next_lane.center_line[0].y - next_lane.center_line[1].y,
                next_lane.center_line[0].x - next_lane.center_line[1].x,
            )
            ref = self._half_circle(hc_start_point, hc_start_phi, hc_end_point, hc_end_phi)
            x_y_list = []
            for point in ref:
                x_y_dict = {}
                x_y_dict["x"] = point[0]
                x_y_dict["y"] = point[1]
                x_y_list.append(x_y_dict)
        except Exception as e:
            pass

    def sort_json(self, json_obj):
        # sort json file to avoid change after each convert
        # note: avoid sort data which has order
        json_obj["lane2junction"] = sorted(json_obj["lane2junction"])
        json_obj["junction2lane"] = sorted(json_obj["junction2lane"])
        json_obj["junctionConnections"] = sorted(json_obj["junctionConnections"], key=lambda x: x["connectionId"])
        json_obj["lanes"] = sorted(json_obj["lanes"], key=lambda x: x["laneId"])
        return json_obj

    def save(self, path):
        strjson = json_format.MessageToJson(self.pb_attach_map, indent=4)
        json_obj = json.loads(strjson)
        json_obj = self.sort_json(json_obj)
        strjson = json.dumps(json_obj, indent=4)
        with open(path, "w") as f:
            f.write(strjson)

    def lane_acc_length(self, lane):
        l_list = [0]
        single_length = 0
        for i in range(len(lane.center_line) - 1):
            single_length += math.sqrt(
                math.pow(lane.center_line[i + 1].x - lane.center_line[i].x, 2)
                + math.pow(lane.center_line[i + 1].y - lane.center_line[i].y, 2)
            )
            l_list.append(single_length)
        return l_list

    def _get_last_5m_point(self, lane):
        if len(lane.center_line) < 2:
            return []
        else:
            s_list = self.lane_acc_length(lane)
            if len(s_list) == 0:
                return []
            elif s_list[-1] <= 5:
                point_list = []
                for p in lane.center_line:
                    point_list.append([p.x, p.y])
                return point_list
            else:
                total_length = s_list[-1]
                s_interp = [total_length - 0.5 * ii for ii in range(10)]
                x_list = []
                y_list = []
                for p in lane.center_line:
                    x_list.append(p.x)
                    y_list.append(p.y)
                x_interp = np.interp(s_interp, s_list, x_list)
                y_interp = np.interp(s_interp, s_list, y_list)
                last_5m_point = []
                assert len(x_interp) == len(y_interp)
                for x_, y_ in zip(x_interp, y_interp):
                    last_5m_point.append([x_, y_])
            return last_5m_point

    def _add_stop_line_to_lane(self, stop_line, lane):
        lane_id = lane.lane_id
        # print(stop_line.stopline_id, stop_line.stopline_id)
        for lane_attach in self.pb_attach_map.lanes:
            if lane_attach.lane_id == lane_id:
                lane_attach.stop_line.has_stop_line = True
                lane_attach.stop_line.stop_line_id = stop_line.stopline_id
                for p in stop_line.shape:
                    pa = hdmap_attach_pb2.Point()
                    pa.x = p.x
                    pa.y = p.y
                    lane_attach.stop_line.shape.append(pa)
                break

    def _add_stop_line_to_jc(self, stop_line, jc):
        # print(stop_line.stopline_id, jc.connection_id)
        jc.stop_line.has_stop_line = True
        jc.stop_line.stop_line_id = stop_line.stopline_id
        for p in stop_line.shape:
            pa = hdmap_attach_pb2.Point()
            pa.x = p.x
            pa.y = p.y
            jc.stop_line.shape.append(pa)

    def match_stop_line(self):
        for lane in self.map_pb.lanes:
            last_5m_point = self._get_last_5m_point(lane)
            for stop_line in self.map_pb.stoplines:
                for point1 in last_5m_point:
                    match1 = False
                    for point2 in stop_line.shape:
                        dist = math.sqrt(math.pow(point1[0] - point2.x, 2) + math.pow(point1[1] - point2.y, 2))
                        if dist < 5:
                            match1 = True
                            self._add_stop_line_to_lane(stop_line, lane)
                            break
                    if match1:
                        break

        for jc in self.pb_attach_map.junction_connections:
            last_5m_point = self._get_last_5m_point(jc)
            for stop_line in self.map_pb.stoplines:
                for point1 in last_5m_point:
                    match1 = False
                    for point2 in stop_line.shape:
                        dist = math.sqrt(math.pow(point1[0] - point2.x, 2) + math.pow(point1[1] - point2.y, 2))
                        if dist < 5:
                            match1 = True
                            self._add_stop_line_to_jc(stop_line, jc)
                            break
                    if match1:
                        break
