import numpy as np
import math
from gops.utils.math_utils import convert_ground_coord_to_ego_coord
import functools
import time

def inverse_normalize_action(action: np.array, 
                             action_half_range: np.array,
                             action_center: np.array) -> np.array:
    action = action * action_half_range + action_center
    return action

def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def cal_dist(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

# get the indices of the smallest k elements
def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr, k)
    return idx[:k]

def calculate_perpendicular_points(x0, y0, direction_radians, distance):
    dx = -math.sin(direction_radians)
    dy = math.cos(direction_radians)

    x1 = x0 + distance * dx
    y1 = y0 + distance * dy
    x2 = x0 - distance * dx
    y2 = y0 - distance * dy

    return (x1, y1), (x2, y2)

# def coordinate_transformation(x_0, y_0, phi_0, x, y, phi):
#     x_ = (x - float(x_0)) * np.cos(float(phi_0)) + (y - float(y_0)) * np.sin(float(phi_0))
#     y_ = -(x - x_0) * np.sin(phi_0) + (y - y_0) * np.cos(phi_0)
#     phi_ = phi - phi_0
#     return np.array((x_, y_, phi_), axis=-1)

def timeit(func):
    """Decorator to measure the execution time of a method."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # Start the timer
        value = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        # print(f"Function {func.__name__!r} took {elapsed_time:.4f} seconds to complete.")
        return value

    return wrapper_timer