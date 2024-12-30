import random
import json

import numpy as np
from matplotlib import cm as cmx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import gops.utils.map_tool.proto.hdmap_pb2 as hdmap_pb2
import google.protobuf.text_format as text_format
from google.protobuf import json_format

from gops.utils.map_tool.lib.map_base import MapBase


class Map(MapBase):
    def __init__(self) -> None:
        super().__init__()
        self.colors = []
        self.init_colors()
        self.map_name = "map"

    def load(self, map_file_name, attach_map_name=None):
        super().load(map_file_name)
    #     if attach_map_name is not None:
    #         with open(attach_map_name, "r", encoding="utf-8") as f:
    #             json_obj = json.load(f)

    #         json_str = json.dumps(json_obj, indent=4)

    def load_hd(self, traffic_map):
        self.map_pb = traffic_map.data
        json_obj = json_format.MessageToDict(self.map_pb,preserving_proto_field_name=True)
        print(json_obj.keys())
        # file_path = '/home/idlab/code/qx-oracle/data_qx/rl_planner/output.json'
        # with open(file_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_obj, f, indent=4)
        self.map_pb = {}
        self.map_pb["links"] = []
        self.map_pb["lanes"] = []
        self.map_pb["junctions"] = []
        self.map_pb["segments"] = []
        self.map_pb["stop_lines"] = []
        self.map_pb["connections"] = []
        link_index = 0  
        lane_index = 0
        for index, segment in enumerate(json_obj["segments"]):
            self.map_pb["segments"].append(segment)
            self.segment2idx[segment["id"]] = index
            self.segment_id_list.append(segment["id"])

            for link in segment["ordered_links"]:
                link["segment_id"] = segment["id"]
                self.link2idx[link["id"]] = link_index
                link_index += 1
                self.link_id_list.append(link["id"])
                
                self.map_pb["links"].append(link)
                
                for lane in link["ordered_lanes"]:
                    self.lane2idx[lane["id"]] = lane_index
                    lane_index += 1
                    self.lane_id_list.append(lane["id"])
                    self.map_pb["lanes"].append(lane)
                    stopline = lane.get("stopline")
                    if stopline is not None:
                        self.map_pb["stop_lines"].append(stopline)

        for index,junction in enumerate(json_obj["junctions"]):
            self.junction2idx[junction["id"]] = index
            self.junction_id_list.append(junction["id"])
            self.map_pb["segments"].append(segment)
            for _, connection in enumerate(junction.get("connections", [])):
                self.map_pb["connections"].append(connection)

    def init_colors(self):
        color_num = 6
        self.colors = []
        values = list(range(color_num))
        jet = plt.get_cmap("brg")
        color_norm = mcolors.Normalize(vmin=0, vmax=values[-1])
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=jet)
        for val in values:
            color_val = scalar_map.to_rgba(val)
            self.colors.append(color_val)

    def draw_lane(self, is_show_lane_ids=False):
        x_range = [np.inf, -np.inf]
        y_range = [np.inf, -np.inf]
        cnt = 1
        for lane in self.map_pb["lanes"]:
            if lane.get("type") == "unknown":
                continue
            color_val = self.colors[cnt % len(self.colors)]

            rx, ry = self._draw_lane_central(lane, color_val)

            x_range[0] = np.min([x_range[0], rx[0]])
            x_range[1] = np.max([x_range[1], rx[1]])
            y_range[0] = np.min([y_range[0], ry[0]])
            y_range[1] = np.max([y_range[1], ry[1]])

            if is_show_lane_ids:
                self._draw_lane_id(lane, color_val)

            cnt += 1
        return x_range, y_range

    def draw_link(self, is_show_link_ids=False):
        cnt = 1
        for link in self.map_pb["links"]:
            color_val = self.colors[cnt % len(self.colors)]
            self._draw_link_boundary(link, color_val)

    def draw_stopline(self, is_show_id=False):
        color_val = (1.0, 0.0, 0.0, 1.0)
        for stopline in self.map_pb["stop_lines"]:
            self._draw_stopline(stopline, color_val)

    def draw_junction_shape(self, is_show_id=False):
        color_val = (0.0, 0.0, 1.0, 1.0)
        for junction in self.map_pb["junctions"]:
            self._draw_junction_shape(junction, color_val)
            if is_show_id:
                self._draw_junction_id(junction, color_val)

    def _draw_lane_id(self, lane, color_val):
        """draw lane id"""
        if (len(lane["center_line"]) == 0):
            return
        x, y = self._find_lane_central_point(lane)
        link_id = lane["link_id"]
        link_idx = self.link2idx[link_id]
        link = self.map_pb["links"][link_idx]
        segment_id = link["segment_id"]

        id_str = "la:" + lane["id"] + "\n" + "li:" + link_id + "\n" + "se:" + segment_id
        # self._draw_label(id_str, (x, y), color_val)

    def _draw_link_id(self, link, color_val):
        x, y = self._find_link_boundary_central_point(link)
        # self._draw_label(link.link_id, (x, y), color_val)

    def _draw_junction_id(self, junction, color_val):
        """draw lane id"""
        px = []
        py = []
        for p in junction["shape"]:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

        if len(px) > 0:
            x = (np.min(px) + np.max(px)) / 2
            y = (np.min(py) + np.max(py)) / 2

            junction_id = junction["junction_id"]
            id_str = "junc:" + junction_id
            # self._draw_label(id_str, (x, y), color_val)

    @staticmethod
    def _find_lane_central_point(lane):
        cp_idx = len(lane["center_line"]) // 2
        # print(len(lane.center_line), cp_idx, lane.lane_id)
        cp = lane["center_line"][cp_idx]
        x = cp.get("point", {}).get("x", 0)
        y = cp.get("point", {}).get("y", 0)
        return x, y

    @staticmethod
    def _find_link_boundary_central_point(link):
        cp_idx = len(link.left_boundary) // 2
        # print(len(lane.center_line), cp_idx, lane.lane_id)
        cp = link["left_boundary"][cp_idx]

        return cp["point"]["x"], cp["point"]["y"]

    @staticmethod
    def _draw_label(label_id, point, color_val):
        """draw label id"""
        labelxys = []
        labelxys.append((40, -40))
        labelxys.append((-40, -40))
        labelxys.append((40, 40))
        labelxys.append((-40, 40))
        has = ["right", "left", "right", "left"]
        vas = ["bottom", "bottom", "top", "top"]

        idx = random.randint(0, 3)
        lxy = labelxys[idx]
        plt.annotate(
            label_id,
            xy=(point[0], point[1]),
            xytext=lxy,
            textcoords="offset points",
            ha=has[idx],
            va=vas[idx],
            bbox=dict(boxstyle="round,pad=0.5", fc=color_val, alpha=0.1),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc=color_val, ec=color_val, alpha=0.1),
        )

    @staticmethod
    def _draw_stopline(stopline, color_val):
        px = []
        py = []
        for p in stopline["shape"]["points"]:
            px.append(float(p.get("x", 0)))
            py.append(float(p.get("y", 0)))

        plt.plot(px, py, ls="-", c=color_val, alpha=0.5, linewidth=5)

    @staticmethod
    def _draw_junction_shape(junction, color_val):
        px = []
        py = []
        for p in junction["shape"]:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

        for p in junction["shape"]:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

            break
        plt.plot(px, py, ls="-", c=color_val, alpha=0.5)

    @staticmethod
    def _draw_link_boundary(link, color_val):
        px = []
        py = []
        for p in link["left_boundary"]:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

        plt.plot(px, py, ls="-", c=color_val, alpha=0.5)

        px = []
        py = []
        for p in link.right_boundary:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

        plt.plot(px, py, ls="-", c=color_val, alpha=0.5)

    @staticmethod
    def _draw_lane_central(lane, color_val):
        """draw boundary"""
        x_range = [np.inf, -np.inf]
        y_range = [np.inf, -np.inf]

        px = []
        py = []
        for p in lane["center_line"]:
            px.append(float(p["point"].get("x", 0)))
            py.append(float(p["point"].get("y", 0)))

        if len(px) > 0:
            x_range[0] = np.min([x_range[0], np.min(np.array(px))])
            x_range[1] = np.max([x_range[1], np.max(np.array(px))])
            y_range[0] = np.min([y_range[0], np.min(np.array(py))])
            y_range[1] = np.max([y_range[1], np.max(np.array(py))])
        # plt.plot(px, py, ls=":", c=color_val, alpha=1, linewidth=2)
        plt.plot(px, py, ls="-", c=color_val, alpha=0.8, linewidth=2)
        return x_range, y_range

    def _draw_attach_lane_id(self, jc, color_val):
        """draw lane id"""
        x, y = self._find_lane_central_point(jc)
        if jc.stop_line.has_stop_line:
            id_str = "sl:" + jc.stop_line.stop_line_id
        # self._draw_label(id_str, (x, y), color_val)

    def draw_connections(self):
        for connection in self.map_pb["connections"]:
            px = []
            py = []
            for p in connection["path"]["points"]:
                px.append(float(p.get("x", 0)))
                py.append(float(p.get("y", 0)))

            # plt.plot(px, py, ls=":", alpha=1, linewidth=2)
            plt.plot(px, py, ls="-", alpha=0.8, linewidth=2)

    def draw(self, show_id=False, show_link_boundary=False, map_name="map"):
        x_range, y_range = self.draw_lane(show_id)
        if show_link_boundary:
            self.draw_link(show_id)
        self.draw_stopline()
        self.draw_junction_shape(show_id)

        # self.draw_attach_lane()
        self.draw_connections()

        fig = plt.gcf()
        ax = plt.gca()
        dpi = fig.get_dpi()
        scale = 0.6

        # Image size must be less than 2^16 in each direction.
        max_width_height = 2 ** 16
        if (x_range[1] - x_range[0]) * scale * dpi > max_width_height:
            scale = max_width_height / (x_range[1] - x_range[0]) / dpi - 0.01
        if (y_range[1] - y_range[0]) * scale * dpi > max_width_height:
            scale = max_width_height / (y_range[1] - y_range[0]) / dpi - 0.01
        width = (x_range[1] - x_range[0]) * scale
        height = (y_range[1] - y_range[0]) * scale

        print(f"The map size is: {width}, {height}")
        # fig.set_figwidth(width)
        # fig.set_figheight(height)

        plt.axis("equal")
        plt.savefig("./" + map_name + ".pdf", dpi=2000)
        # plt.show()

    def draw_everything(self, show_id=False, show_link_boundary=False, map_name="map"):
        x_range, y_range = self.draw_lane(show_id)
        if show_link_boundary:
            self.draw_link(show_id)
        self.draw_stopline()
        self.draw_junction_shape(show_id)

        # self.draw_attach_lane()
        self.draw_connections()

        fig = plt.gcf()
        ax = plt.gca()
        dpi = fig.get_dpi()
        scale = 0.6

        # Image size must be less than 2^16 in each direction.
        max_width_height = 2 ** 16
        if (x_range[1] - x_range[0]) * scale * dpi > max_width_height:
            scale = max_width_height / (x_range[1] - x_range[0]) / dpi - 0.01
        if (y_range[1] - y_range[0]) * scale * dpi > max_width_height:
            scale = max_width_height / (y_range[1] - y_range[0]) / dpi - 0.01
        width = (x_range[1] - x_range[0]) * scale
        height = (y_range[1] - y_range[0]) * scale

        # fig.set_figwidth(width)
        # fig.set_figheight(height)
        plt.axis("equal")
        plt.savefig("./" + map_name + ".pdf", dpi=2000)

