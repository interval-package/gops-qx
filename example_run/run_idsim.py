from gops.sys_simulator.qxsys_run import PolicyRunner
from gops.env.env_gen_ocp.resources.idsim_model.params import qianxing_config
from gops.env.env_gen_ocp.resources.idsim_model.utils.generate_gif import process_batch

if __name__ == "__main__":

    qianxing_config["render_flag"] = True

    policy = "/home/zhengziang/code/gops-qx/results/pyth_idsim/DSACTPI_241012-143200"
    model = "500000"
    policy_name = policy.split("/")[-1]
    policy_idxs = [0]


    runner = PolicyRunner(
        log_policy_dir_list=[policy],  ##results/pyth_idsim/DSACTPI_240906-110836/apprfunc/apprfunc_3000.pkl
        trained_policy_iteration_list=[model],  # results/pyth_idsim/DSACTPI_240906-114753/apprfunc/apprfunc_61000.pkl
        is_init_info=False, #DSACTPI_240906-141025 DSACTPI_240906-114753
        # init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
        save_render=False,
        legend_list=["None"],
        use_opt=False, # Use optimal solution for comparison
        # opt_args={
        #     "opt_controller_type": "MPC", # MPC or OPT
        #     "num_pred_step": 80,
        #     "gamma": 1,
        #     "minimize_options": {"max_iter": 200, "tol": 1e-3,
        #                          "acceptable_tol": 1e0,
        #                          "acceptable_iter": 10,},
        #     "use_terminal_cost": False,
        # },
        dt=None,  # time interval between steps
    )

    for pidx in policy_idxs:
        run_config = {
            "policy": policy_name,
            "draw_bound": 30,
            "show_npc": False,
            "task_id": runner.args_list[pidx]["qianxingp_task_id"]
        }

        qianxing_config["render_info"].update(run_config)

        runner.run_single(pidx)

        if "path" in qianxing_config["render_info"].keys():
            path = qianxing_config["render_info"]["path"]
            print(qianxing_config["render_info"])

            process_batch(path, f"iter_{model}")