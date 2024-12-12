from argparse import ArgumentParser
import json
import os
from gops.sys_simulator.qxsys_run import PolicyRunner
from gops.env.env_gen_ocp.resources.idsim_model.params import qianxing_config
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_gif import process_batch
from gops.env.env_gen_ocp.resources.idsim_model.utils.vedio_utils.generate_vedio import gen_video_cv2_mp4, gen_video_imio_mp4

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./example_run/draw.json")
    config = parser.parse_known_args()[0].config
    if config is None:
        parser.add_argument("--load", type=str)
        parser.add_argument("--ckpt", type=str)
        return vars(parser.parse_args())
    else:
        with open(config, "rt") as f:
            args = json.load(f)
        return args

if __name__ == "__main__":

    args = parse_args()
    qianxing_config["render_flag"] = True
    qianxing_config["traj_flag"] = True

    policy      = [args["load"]]
    models       = [args["ckpt"]]
    
    policy_idxs = [0]

    runner = PolicyRunner(
        log_policy_dir_list=policy, 
        trained_policy_iteration_list=models, 
        is_init_info=False, 
        save_render=False,
        legend_list=["None"],
        use_opt=False, 
        dt=None, 
    )

    for pidx in policy_idxs:
        print(runner.args_list[pidx]["qianxingp_task_id"])
        policy_name = policy[pidx].split("/")[-1]
        run_config = {
            "policy": policy_name,
            "draw_bound": 30,
            "show_npc": False,
            "_debug_path_qxdata": None,
        }

        qianxing_config["render_info"].update(run_config)
        qianxing_config["task_id"] = runner.args_list[pidx]["qianxingp_task_id"]

        # runner.args_list[pidx]["qx_config"].update(qianxing_config)
        runner.args_list[pidx].update({
            "max_steps": 500,
            "qx_config": None
        })

        runner.single_run(pidx)

        if "_debug_path_qxdata" in qianxing_config["render_info"].keys():
            path = qianxing_config["render_info"]["_debug_path_qxdata"]
            print(qianxing_config["render_info"])
            fname = f"mdl_{models[pidx]}.mp4"
            # process_batch(path, fname)
            gen_video_imio_mp4(path, os.path.join(path, fname))