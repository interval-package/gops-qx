import argparse
import json

def parse_piconifg(parser:argparse.ArgumentParser, base_env_config, base_env_model_config, max_iter, cal_idsim_pi_paras):
    pi_paras = cal_idsim_pi_paras(env_config=base_env_config, env_model_config=base_env_model_config)
    parser.add_argument("--target_PI", type=bool, default=True)
    parser.add_argument("--enable_self_attention", type=bool, default=False)
    parser.add_argument("--pi_begin", type=int, default=pi_paras["pi_begin"])
    parser.add_argument("--pi_end", type=int, default=pi_paras["pi_end"])
    parser.add_argument("--enable_mask", type=bool, default=True)
    parser.add_argument("--obj_dim", type=int, default=pi_paras["obj_dim"])
    parser.add_argument("--attn_dim", type=int, default=64)
    parser.add_argument("--pi_out_dim", type=int, default=pi_paras["output_dim"])
    parser.add_argument("--pi_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument("--pi_hidden_activation", type=str, default="gelu")
    parser.add_argument("--pi_output_activation", type=str, default="linear")
    parser.add_argument("--freeze_pi_net", type=str, default="critic")
    parser.add_argument("--encoding_others", type=bool, default=False)
    parser.add_argument("--others_hidden_sizes", type=list, default=[64,64])
    parser.add_argument("--others_hidden_activation", type=str, default="gelu")
    parser.add_argument("--others_output_activation", type=str, default="linear")
    parser.add_argument("--others_out_dim", type=int, default=32)
    parser.add_argument("--policy_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })

    parser.add_argument("--q1_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })
    parser.add_argument("--q2_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })
    parser.add_argument("--pi_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })

    parser.add_argument("--alpha_scheduler", type=json.loads, default={
        "name": "CosineAnnealingLR",
        "params": {
                "T_max": max_iter,
            }
    })