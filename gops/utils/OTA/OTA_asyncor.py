import importlib
import contextlib
import os, sys
import json
import requests
import argparse
import torch
import onnxruntime as ort
from gops.utils.OTA.onnx.dsac import onnx_dsac
from typing import Tuple
import subprocess

from gops.utils.OTA.onnx.pkl2onnx_dsact import \
    DSAC_policy_export_onnx_model, onnx_brute_force, _load_args

onnx_script = "gops/utils/OTA/onnx/pkl2onnx_dsact.py"

"""
Only upload policy
"""

class OTA_asyncor:
    def __init__(self, ckpt_dir, host="localhost", port=2790, **kwargs):
        self.ckpt_dir = ckpt_dir
        self.model_dir, self.traj_dir, self.onnx_dir = self.util_make_dir(ckpt_dir)

        # Create directories if they don't exist
        os.makedirs(self.onnx_dir, exist_ok=True)
        os.makedirs(self.traj_dir, exist_ok=True)
        
        # parsing config
        config_file = os.path.join(ckpt_dir, "config.json")
        # with open(config_file) as f:
        #     config = json.load(fp=f)
        self.config = _load_args(ckpt_dir)
        self.parse_model_config()

        # Server details
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def parse_model_config(self):
        args = self.config
        obs_dim = args["obsv_dim"]
        act_dim = args["action_dim"]
        ego_dim = args["env_model_config"]["ego_feat_dim"]
        sur_dim = args["env_model_config"]["per_sur_feat_dim"] + 3 # +3 for length, width, mask
        ref_dim = args["env_model_config"]["per_ref_feat_dim"]

        num_ref_points = len(args["env_model_config"]["downsample_ref_point_index"])
        num_objs = int(sum(i for i in args["env_config"]["obs_num_surrounding_vehicles"].values()))
        obs_dict = {
            "ego_dim": ego_dim,
            "sur_dim": sur_dim,
            "ref_dim": ref_dim,
            "num_objs": num_objs,
            "num_ref_points": num_ref_points,
        }
        action_upper_bound = args["env_config"]["action_upper_bound"]
        action_lower_bound = args["env_config"]["action_lower_bound"]    
        action_scale_factor = 1
        action_scale_bias = 0
        obs_scale_factor = args["obs_scale"]
        obs_scale_factor = torch.tensor(obs_scale_factor).float()
        cfg = {
            "input_dim": obs_dim,
            "act_scale_factor": action_scale_factor,
            "obs_scale_factor": obs_scale_factor,
            "bias": action_scale_bias,
            "obs_config": obs_dict
        }
        self.onnx_cfg = cfg

        alg_file_name = args["algorithm"].lower()
        file = importlib.import_module("gops.algorithm." + alg_file_name)
        self.networks = getattr(file, "ApproxContainer")(**args)

    def async_mdl(self, version, iteration=None):
        # local_traj_list = self.util_get_file_list(self.traj_dir, ".csv")
        local_mdl_list = self.util_get_file_list(self.model_dir, ".pkl")
        assert local_mdl_list, "The mdls are empty"
        liter, _ = self.util_get_opt(local_mdl_list, self.util_parse_apprname)
        success, download_info = self.get_model_version()        
        mdl_version = version
        mdl_path = os.path.join(self.model_dir, f"apprfunc_{liter}.pkl")
        onnx_path = os.path.join(self.onnx_dir, f"{mdl_version}.onnx")

        print(f"Onnxlize the {mdl_path} to {onnx_path}...")
        result = subprocess.run(['python', onnx_script, '--ckpt', self.ckpt_dir, '--source', mdl_path, '--target', onnx_path], check=True)
        print("Script finished with exit code:", result.returncode)
        
        success, upload_info = self.upload_model(onnx_path)
        if success:
            print("Successfully uploading.")
        else:
            print(f"Fail to upload, report: {upload_info}")
        return

    def async_traj(self, version:str=None, idx:int=None):
        """
        traj named after mdlversion(x.x.x).x.csv
        """
        local_traj_list = self.util_get_file_list(self.traj_dir, ".csv")
        return

    def local_get_opt_mdl(self):
        local_mdl_list = self.util_get_file_list(self.model_dir, ".pkl")
        miter, optiter = self.util_get_opt(local_mdl_list, self.util_parse_apprname)
        return os.path.join(self.model_dir, f"apprfunc_{miter if optiter < 0 else f'{optiter}_opt'}.pkl")

    ## Comunicate

    def upload_model(self, model_path: str) -> Tuple[bool, str]:
        """
        Uploads the specified ONNX model file to the server.
        Returns a tuple (success, message) indicating the result of the upload.
        """
        if not os.path.exists(model_path):
            return False, f"Model file {model_path} does not exist."

        url = f"{self.base_url}/upload/model"
        files = {'file': open(model_path, 'rb')}
        
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                return True, "Model uploaded successfully."
            else:
                return False, f"Failed to upload model: {response.text}"
        except Exception as e:
            return False, str(e)

    def upload_traj(self, traj_path: str) -> Tuple[bool, str]:
        """
        Uploads the specified trajectory file to the server.
        Returns a tuple (success, message) indicating the result of the upload.
        """
        if not os.path.exists(traj_path):
            return False, f"Trajectory file {traj_path} does not exist."

        url = f"{self.base_url}/upload/traj"
        files = {'file': open(traj_path, 'rb')}
        
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                return True, "Trajectory uploaded successfully."
            else:
                return False, f"Failed to upload trajectory: {response.text}"
        except Exception as e:
            return False, str(e)

    def download_model(self, version: str) -> Tuple[bool, str]:
        """
        Downloads the specified model version from the server.
        Saves it in the ONNX directory.
        """
        url = f"{self.base_url}/download/model/{version}"
        model_save_path = os.path.join(self.onnx_dir, f"{version}.onnx")

        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(model_save_path, 'wb') as f:
                    f.write(response.content)
                return True, f"Model version {version} downloaded successfully."
            else:
                return False, f"Failed to download model: {response.text}"
        except Exception as e:
            return False, str(e)

    def download_traj(self, version: str) -> Tuple[bool, str]:
        """
        Downloads the specified trajectory version from the server.
        Saves it in the trajectory directory.
        """
        url = f"{self.base_url}/download/traj/{version}"
        traj_save_path = os.path.join(self.traj_dir, f"{version}.csv")

        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(traj_save_path, 'wb') as f:
                    f.write(response.content)
                return True, f"Trajectory version {version} downloaded successfully."
            else:
                return False, f"Failed to download trajectory: {response.text}"
        except Exception as e:
            return False, str(e)

    def get_model_version(self) -> Tuple[bool, str]:
        """
        Retrieves the current model version from the server.
        """
        url = f"{self.base_url}/info/mdlversion"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True, response.text
            else:
                return False, f"Failed to retrieve model version: {response.text}"
        except Exception as e:
            return False, str(e)

    def get_traj_version(self) -> Tuple[bool, str]:
        """
        Retrieves the current trajectory version from the server.
        """
        url = f"{self.base_url}/info/trajversion"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True, response.text
            else:
                return False, f"Failed to retrieve trajectory version: {response.text}"
        except Exception as e:
            return False, str(e)

    def list_models(self) -> Tuple[bool, str]:
        """
        Retrieves the list of available models from the server.
        """
        url = f"{self.base_url}/info/mdllist"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Failed to retrieve model list: {response.text}"
        except Exception as e:
            return False, str(e)

    def list_trajs(self) -> Tuple[bool, str]:
        """
        Retrieves the list of available trajectories from the server.
        """
        url = f"{self.base_url}/info/trajlist"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Failed to retrieve trajectory list: {response.text}"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def util_get_file_list(folder_path, extension="pkl"):
        return [f for f in os.listdir(folder_path) if f.endswith(extension)]

    @staticmethod
    def util_make_dir(ckpt):
        return os.path.join(ckpt, "apprfunc"), os.path.join(ckpt, "traj"), os.path.join(ckpt, "onnx")
    
    @staticmethod
    def util_parse_apprname(apprname: str) -> Tuple[int, bool]:
        """
        Parses the filename to extract iteration and optimization status.
        Example: apprname -> 'model_12opt.onnx' -> iteration: 12, is_opt: True
        """
        apprname = apprname[:-4].split("_")[1]
        is_opt = apprname[-3:] == "opt"
        iteration = int(apprname[:-3] if is_opt else apprname)
        return iteration, is_opt

    @staticmethod
    def util_get_opt(file_list, parse_func):
        miter = -1
        optiter = -1
        for fname in file_list:
            iter, is_opt = parse_func(fname)
            optiter = iter if is_opt else optiter
            miter = max(iter, miter)
        return miter, optiter

def OTA_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5000)
    parser.add_argument("--ckpt_dir", type=str)
    args = parser.parse_args()
    assert os.path.exists(args.ckpt_dir), "No such info saved."
    return args

def main():
    # args = OTA_parser()
    # ckpt = args.ckpt
    ckpt_dir = "results/pyth_idsim/DSACTPI_241013-023014"
    args_debug = {
        "host": "localhost",
        "port": 2790,
        "ckpt_dir": ckpt_dir,
    }
    obj = OTA_asyncor(**args_debug)
    obj.async_mdl(version="1.0.1")
    return


if __name__== '__main__':
    main()
    pass
