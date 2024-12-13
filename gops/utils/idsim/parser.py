import argparse, json
from gops.env.env_gen_ocp.resources.idsim_config_multilane import \
    get_idsim_env_config, get_idsim_model_config, cal_idsim_obs_scale, cal_idsim_pi_paras


def parse_idsim_args(parser:argparse.ArgumentParser):
    parser.add_argument("--env_scenario", type=str, default="multilane", help="crossroad / multilane")
    parser.add_argument("--num_threads_main", type=int, default=4, help="Number of threads in main process")
    env_scenario = parser.parse_known_args()[0].env_scenario

    # Params for idsim env
    base_env_config = get_idsim_env_config(env_scenario)
    base_env_model_config = get_idsim_model_config(env_scenario)

    parser.add_argument("--extra_env_config", type=str, default=r'{}')
    extra_env_config = parser.parse_known_args()[0].extra_env_config
    extra_env_config = json.loads(extra_env_config)

    parser.add_argument("--extra_env_model_config", type=str, default=r'{}')
    extra_env_model_config = parser.parse_known_args()[0].extra_env_model_config
    extra_env_model_config = json.loads(extra_env_model_config)

    base_env_config.update(extra_env_config)
    base_env_model_config.update(extra_env_model_config)

    parser.add_argument("--env_config", type=dict, default=base_env_config)
    parser.add_argument("--env_model_config", type=dict, default=base_env_model_config)

    parser.add_argument("--scenerios_list", type=list, default=[':19','19:'])

    parser.add_argument("--vector_env_num", type=int, default=4, help="Number of vector envs")
    parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    parser.add_argument("--gym2gymnasium", type=bool, default=True, help="Convert Gym-style env to Gymnasium-style")

    parser.add_argument("--ego_scale", type=list, default=[1, 20, 20, 1, 4, 1, 4] ) #  vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    parser.add_argument("--sur_scale", type=list, default=[0.2, 1, 1, 10, 1, 1, 1, 1] ) #  rel_x, rel_y , cos(phi), sin(phi), speed, length, width, mask
    parser.add_argument("--ref_scale", type=list, default=[0.2, 1, 1, 10, 1] ) # ref_x ref_y ref_cos(ref_phi) ref_sin(ref_phi), error_v
    ego_scale = parser.parse_known_args()[0].ego_scale
    sur_scale = parser.parse_known_args()[0].sur_scale
    ref_scale = parser.parse_known_args()[0].ref_scale
    obs_scale = cal_idsim_obs_scale(
        ego_scale=ego_scale,
        sur_scale=sur_scale,
        ref_scale=ref_scale,
        env_config=base_env_config,
        env_model_config=base_env_model_config
    )
    parser.add_argument("--obs_scale", type=dict, default=obs_scale)
    parser.add_argument("--repeat_num", type=int, default=4, help="action repeat num")

    parser.add_argument("--qianxingp_task_id", type=int, default=82, help="Qianxing task id")
    parser.add_argument("--qianxingp_token", type=int, default=None, help="Qianxing task id")
    return base_env_config, base_env_model_config