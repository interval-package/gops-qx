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


def init_args()->Tuple[RenderCfg,str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, default="./example_run/recover.json")
    args = parser.parse_args()
    ckpt:str = args.ckpt
    if ckpt.endswith(".json"):
        with open(ckpt, "rt") as f:
            ckpt = json.load(f)["ckpt"]

    ckpt_cfg = os.path.join(ckpt, "render_config.pkl")
    ckpt_traj = os.path.join(ckpt, "trajs.pkl")
    with open(ckpt_cfg, "rb") as f:
        cfg:RenderCfg = pickle.load(f)
    return cfg, ckpt_traj

def main():
    cfg, ckpt_traj = init_args()

    # draw map
    f = plt.figure(figsize=(16,9))
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
        LasvsimEnv._render_sur_byobs(context, show_debug=True, show_done=False)
        pass

    return 

if __name__ == "__main__":
    main()
    pass