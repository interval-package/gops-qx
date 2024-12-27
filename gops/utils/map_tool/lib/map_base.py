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


class MapBase:
    def __init__(self) -> None:
        self.map_pb = hdmap_pb2.HdMap()

        self.lane2idx = dict()
        self.link2idx = dict()
        self.segment2idx = dict()
        self.junction2idx = dict()

        self.lane_id_list = []
        self.link_id_list = []
        self.segment_id_list = []
        self.junction_id_list = []

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

    def load_new(self,map_file_name):
        with open(map_file_name, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
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
