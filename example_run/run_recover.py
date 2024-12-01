from collections import deque
import os, argparse
import pickle
import json
from typing import Tuple

from matplotlib import pyplot as plt
from gops.env.env_gen_ocp.resources.idsim_model.lasvsim_env_qianxing import LasvsimEnv
from gops.env.env_gen_ocp.resources.idsim_model.utils.las_render import \
    LasStateSurrogate, RenderCfg, render_tags_debug,\
    load_from_pickle_iterable
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_gif import process_batch

def _disp_done(surrogate: LasStateSurrogate):
    print(surrogate._render_done_info)

ref_ax, ref_ay = [], []

def _disp_ref(surrogate: LasStateSurrogate, idx=None):
    print('#'*50)
    msg = f"iter: {idx}" if idx is not None else ""
    print('ref point ')
    for i in range(5):
        ref_x, ref_y = surrogate._ref_points[i, :2].squeeze().tolist()
        msg += f"[x: {ref_x:.2f}, y: {ref_y:.2f}]"
        ref_ax.append(ref_x)
        ref_ay.append(ref_y)
    print(msg)

_draw_debug_done = lambda surrogate: LasvsimEnv._render_sur_byobs(surrogate, show_debug=False, show_done=True)

def init_args()->Tuple[RenderCfg,str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./example_run/recover.json")
    args = parser.parse_args()
    ckpt:str = args.ckpt
    if ckpt.endswith(".json"):
        with open(ckpt, "rt") as f:
            ckpt = json.load(f)["ckpt"]

    ckpt_cfg = os.path.join(ckpt, "render_config.pkl")
    ckpt_traj = os.path.join(ckpt, "trajs.pkl")
    with open(ckpt_cfg, "rb") as f:
        cfg:RenderCfg = pickle.load(f)
    return cfg, ckpt_traj, ckpt

def main():
    cfg, ckpt_traj, path = init_args()

    # draw map
    f = plt.figure(figsize=(16,9), dpi=40)
    width, height = 16*40, 9*40

    f.subplots_adjust(left=0.25)
    cfg.map.draw_everything(show_id=False, show_link_boundary=False)

    shadow = deque([])

    for idx, context in enumerate(load_from_pickle_iterable(ckpt_traj)):
        print(f"frame: {idx}")
        context._render_tags_debug = render_tags_debug
        context.traj_flag = False
        context.render_flag = True
        context._render_cfg = cfg
        context._render_count = idx-1
        context._render_ego_shadows = shadow
        # draw each moment
        # LasvsimEnv._render_sur_byobs(context, show_debug=False, show_done=True)
        # _disp_done(context)
        # _draw_debug_done(context)
        _disp_ref(context)
        pass

    fname = f"res.mp4"
    # process_batch(path, fname)
    # create_video_from_images(path, os.path.join(path, fname))

    return 

if __name__ == "__main__":
    main()
    pass