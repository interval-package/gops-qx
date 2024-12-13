import math
from typing import Union

import numpy as np
import torch


def angle_normalize(
    x: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def deal_with_phi_rad(phi: float):
    return (phi + math.pi) % (2 * math.pi) - math.pi

def convert_ground_coord_to_ego_coord(x, y, phi, ego_x, ego_y, ego_phi):
    shift_x, shift_y = shift(x, y, ego_x, ego_y)
    x_ego_coord, y_ego_coord, phi_ego_coord \
        = rotate(shift_x, shift_y, phi, ego_phi)
    return x_ego_coord, y_ego_coord, phi_ego_coord

def convert_ref_to_ego_coord(ref_obs_absolute, ego_state): # rewrite this function in numpy
    ref_x, ref_y, ref_phi = ref_obs_absolute[:,0], ref_obs_absolute[:,1], ref_obs_absolute[:,2]
    x, y, phi = ego_state[0], ego_state[1], ego_state[4]
    ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = convert_ground_coord_to_ego_coord(
        ref_x, ref_y, ref_phi, x, y, phi)
    return ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord

def shift(orig_x, orig_y, shift_x, shift_y):
    shifted_x = orig_x - shift_x
    shifted_y = orig_y - shift_y
    return shifted_x, shifted_y


def rotate(orig_x, orig_y, orig_phi, rotate_phi):
    rotated_x = orig_x * np.cos(rotate_phi) + orig_y * np.sin(rotate_phi)
    rotated_y = -orig_x * np.sin(rotate_phi) + \
                orig_y * np.cos(rotate_phi)
    rotated_phi = deal_with_phi_rad(orig_phi - rotate_phi)
    return rotated_x, rotated_y, rotated_phi
