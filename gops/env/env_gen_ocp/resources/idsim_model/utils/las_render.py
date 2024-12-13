from collections import deque
from dataclasses import dataclass
import time
from typing import Generator
import matplotlib.pyplot as plt
import os

import numpy as np
from gops.utils.map_tool.lib.map import Map
import json

_render_tags = [
        'env_tracking_error', 
        'env_speed_error', 
        'env_delta_phi', 
        # 'category', 
        # 'env_pun2front', 
        # 'env_pun2side', 
        # 'env_pun2space', 
        # 'env_pun2rear', 
        'env_scaled_reward_part1', 
        'env_reward_collision_risk', 
        'env_scaled_pun2front', 
        'env_scaled_pun2side', 
        'env_scaled_pun2space', 
        'env_scaled_pun2rear', 
        'env_scaled_punish_boundary', 
        # 'state', 
        # 'constraint', 
        # 'env_reward_step', 
        # 'env_reward_steering', 
        # 'env_reward_acc_long', 
        # 'env_reward_delta_steer', 
        # 'env_reward_jerk', 
        # 'env_reward_dist_lat', 
        # 'env_reward_vel_long', 
        # 'env_reward_head_ang', 
        # 'env_reward_yaw_rate', 
        'env_scaled_reward_part2', 
        'env_scaled_reward_step', 
        'env_scaled_reward_dist_lat', 
        'env_scaled_reward_vel_long', 
        'env_scaled_reward_head_ang', 
        'env_scaled_reward_yaw_rate', 
        'env_scaled_reward_steering', 
        'env_scaled_reward_acc_long', 
        'env_scaled_reward_delta_steer', 
        'env_scaled_reward_jerk', 
        'total_reward',
        # 'reward_details', 
        # 'reward_comps'
        ]

render_tags_debug = [
    "_debug_done_errlat",
    "_debug_done_errlon",
    "_debug_done_errhead",
    "_debug_done_postype",
    "_debug_reward_scaled_punish_boundary",
]

@dataclass
class RenderCfg:
    draw_bound = 30
    arrow_len = 1
    # ego_shadows = deque([])
    map = None
    show_npc = False
    _debug_path_qxdata:str = None
    render_type = None
    render_config:dict = None

    def set_vals(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                print(f"No such var for {key}")

    def __getitem__(self, key):
        return self.render_config[key]
    
    def save(self):
        with open(os.path.join(self._debug_path_qxdata, "render_config.pkl"), "wb") as f:
            pickle.dump(self, f)

@dataclass
class LasStateSurrogate:
    _state:             np.ndarray
    _ref_points:        np.ndarray
    action:             np.ndarray
    _ego:               np.ndarray
    _render_surcars:    list
    _render_info:       dict
    _render_done_info:  dict

    # adaptive vars
    _debug_adaptive_vars: dict = None
    
    # Debug vars
    _debug_dyn_state:   np.ndarray = None
    # _debug_done_errlat: np.ndarray = None
    # _debug_done_errlon: np.ndarray = None
    # _debug_done_errhead:np.ndarray = None
    # _debug_done_postype:np.ndarray = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self._debug_adaptive_vars.get(key, "Unrecord")

import pickle

# Append an individual data entry
def append_to_pickle_incremental(file_path, data):
    with open(file_path, "ab") as f:  # Append in binary mode
        pickle.dump(data, f)

# Load all individual entries
def load_from_pickle_incremental(file_path):
    data = []
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))  # Load one object at a time
            except EOFError:
                break
    return data

def load_from_pickle_iterable(file_path)->Generator[LasStateSurrogate, None, None]:
    count = 0
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)  # Load one object at a time
                count += 1
            except EOFError:  # End of file
                print(f"Loading finished with total {count} entries.")
                break

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def ax2np(ax=None, fig=None):
    ax = plt.gca() if ax is None else ax
    fig = plt.gcf() if fig is None else fig

    canvas = FigureCanvas(fig)
    canvas.draw()

    # Get the bounding box of the current axes
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # Crop the canvas to the axis area and get image data
    image_data = np.array(canvas.renderer.buffer_rgba())[int(bbox.y0):int(bbox.y1), int(bbox.x0):int(bbox.x1)]
