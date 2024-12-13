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
