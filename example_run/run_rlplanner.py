from argparse import ArgumentParser
import json
import os
from gops.sys_simulator.qxsys_run import PolicyRunner
from gops.env.env_gen_ocp.resources.idsim_model.params import qianxing_config
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_gif import process_batch
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_vedio import gen_video_cv2_mp4



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./example_run/rlplanner.json")
    config = parser.parse_known_args()[0].config
    with open(config, "rt") as f:
        args = json.load(f)
    return args

if __name__ == "__main__":

    args = parse_args()
    # qianxing_config["render_flag"] = True
    # qianxing_config["traj_flag"] = True

    config_ref = args["qx_load"]
    args_ref = PolicyRunner._load_args(config_ref)
    args_ref = PolicyRunner._process_args(args_ref)
    # args_ref = None

    args_ref["qx_config"].update({
        "token": qianxing_config["token"],
        "_debug_path_qxdata": None,
        "render_flag": True,
        "traj_flag": True,
    })

    config_rlp = args["rlplanner_load"]
    args_rlp = PolicyRunner._load_args(config_rlp)
    args_rlp = PolicyRunner._process_args(args_rlp)

    policies      = [args["rlplanner_load"]]
    models        = [args["rlplanner_ckpt"]]
    
    policy_idxs = [0]

    runner = PolicyRunner(
        log_policy_dir_list=policies, 
        trained_policy_iteration_list=models, 
        is_init_info=False, 
        save_render=False,
        legend_list=["None"],
        use_opt=False, 
        dt=None, 
        rlplanner_flag=True
    )

    for pidx in policy_idxs:
        print(args_ref["qianxingp_task_id"])
        policy_name = policies[pidx].split("/")[-1]
        run_config = {
            "policy": policy_name,
            "draw_bound": 30,
            "show_npc": False,
        }

        qianxing_config["render_info"].update(run_config)
        qianxing_config["task_id"] = args_ref["qianxingp_task_id"]

        runner.single_run(pidx, plc_args=args_rlp, env_args=args_ref)

        if "path" in qianxing_config["render_info"].keys():
            path = qianxing_config["render_info"]["path"]
            print(qianxing_config["render_info"])
            fname = f"mdl_{models[pidx]}.mp4"
            process_batch(path, fname)
            gen_video_cv2_mp4(path, os.path.join(path, fname))

