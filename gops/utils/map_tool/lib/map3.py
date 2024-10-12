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
from collections import OrderedDict
import proto.hdmap_pb2 as hdmap_pb2
from lib.map_base import MapBase
from lib.rule import ADD_JC, DELETE_JC
from google.protobuf.descriptor import FieldDescriptor

class Map3(MapBase):
    def __init__(self, map_name) -> None:
        super().__init__()
        self._map_name = map_name

    def load_map(self):
        with open(self._map_name, "r", encoding="utf-8") as f:
            original_json = json.load(f, object_pairs_hook=OrderedDict)
            self._original_order = list(original_json.keys())
            super().load(self._map_name)

    def fill_connection_ids(self):
        # Iterate through all connections
        for connection in self.map_pb.connections:
            # Fill upstream_link_id and downstream_link_id
            if connection.upstream_lane_id:
                upstream_lane = self.get_lane_by_id(connection.upstream_lane_id)
                connection.upstream_link_id = upstream_lane.link_id
            
            if connection.downstream_lane_id:
                downstream_lane = self.get_lane_by_id(connection.downstream_lane_id)
                connection.downstream_link_id = downstream_lane.link_id

    def calculate_flow_direction(self, path):
        if len(path) < 2:
            raise ValueError("Not enough points to determine direction")

        total_angle_change = 0.0
        prev_angle = math.atan2(path[1].y - path[0].y, path[1].x - path[0].x)

        for i in range(2, len(path)):
            current_angle = math.atan2(path[i].y - path[i-1].y, path[i].x - path[i-1].x)
            angle_change = current_angle - prev_angle
            angle_change = (angle_change + 3 * math.pi) % (2 * math.pi) - math.pi
            total_angle_change += angle_change
            prev_angle = current_angle

        if abs(total_angle_change) < math.pi / 6:
            return "straight"
        elif abs(total_angle_change) > 5 * math.pi / 6:
            return "uturn"
        elif total_angle_change < 0:
            return "right"
        else:
            return "left"

    def update_flow_directions(self):
        for connection in self.map_pb.connections:
            if connection.path:
                connection.flow_direction = self.calculate_flow_direction(connection.path)

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
    
    def _add_stopline_to_lane(self, stop_line, lane):
        if len(lane.stopline_id) == 0:
            lane.stopline_id = stop_line.stopline_id
        
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate the perpendicular distance from point (px, py) to the line segment (x1, y1) - (x2, y2)."""
        # Calculate the line segment's lengths
        line_seg_len_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
        
        if line_seg_len_squared == 0:
            # The segment is a single point
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Consider the line extending the segment, parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_seg_len_squared))
        
        # Projection falls on the segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def match_stopline(self):
        for lane in self.map_pb.lanes:
            last_5m_points = self._get_last_5m_point(lane)
            closest_stopline = None
            min_distance = float('inf')

            for stop_line in self.map_pb.stoplines:
                # Assume stop_line.shape has points [p1, p2] defining the line segment
                assert len(stop_line.shape) == 2
                p1, p2 = stop_line.shape[0], stop_line.shape[1]
                for point1 in last_5m_points:
                    # Calculate perpendicular distance to the line segment
                    dist = self.point_to_line_distance(point1[0], point1[1], p1.x, p1.y, p2.x, p2.y)
                    
                    # Check if this stopline is the closest one within 5 meters so far
                    if dist < 2 and dist < min_distance:
                        min_distance = dist
                        closest_stopline = stop_line
            
            # If a close enough stopline was found, link it to the lane
            if closest_stopline:
                self._add_stopline_to_lane(closest_stopline, lane)
                
    def save_to_json(self, output_filename):
        # Convert protobuf to JSON string with preserving default values
        strjson = self.message_to_json_with_defaults(self.map_pb)

        # Load JSON string as an OrderedDict
        json_obj = json.loads(strjson, object_pairs_hook=OrderedDict)

        # Reorder JSON object according to original order
        ordered_json_obj = OrderedDict()
        for key in self._original_order:
            if key in json_obj:
                ordered_json_obj[key] = json_obj[key]

        # Add any new keys that were not in the original JSON
        for key, value in json_obj.items():
            if key not in ordered_json_obj:
                ordered_json_obj[key] = value

        # Convert back to JSON string with sorted keys disabled
        strjson = json.dumps(ordered_json_obj, indent=4)

        # Write to file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(strjson)

    def message_to_json_with_defaults(self, message):
        """Convert a protobuf message to JSON string, including default values."""
        message_dict = json_format.MessageToDict(message, including_default_value_fields=True, preserving_proto_field_name=True)
        return json.dumps(message_dict, indent=4)

    def sort_json(self, obj):
        if isinstance(obj, dict):
            return {k: self.sort_json(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self.sort_json(x) for x in obj]
        else:
            return obj
