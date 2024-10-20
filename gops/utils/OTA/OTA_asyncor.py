import os
import requests
import argparse
from typing import Tuple

class OTA_asyncor:
    def __init__(self, model_dir: str, traj_dir: str, onnx_dir: str, host="localhost", port=2790, **kwargs):
        self.model_dir = model_dir
        self.traj_dir = traj_dir
        self.onnx_dir = onnx_dir

        # Create directories if they don't exist
        os.makedirs(onnx_dir, exist_ok=True)
        os.makedirs(traj_dir, exist_ok=True)
        
        # Server details
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def async_mdl(self, version, iteration):
        local_onnx_list = self.util_get_file_list(self.onnx_dir, ".onnx")
        local_traj_list = self.util_get_file_list(self.traj_dir, ".csv")
        local_mdl_list = self.util_get_file_list(self.model_dir, ".pkl")
        return

    def async_traj(self, version:str=None, idx:int=None):
        """
        traj named after mdlversion(x.x.x).x.csv
        """
        local_traj_list = self.util_get_file_list(self.traj_dir, ".csv")
        return

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

def OTA_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5000)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    assert os.path.exists(args.ckpt), "No such info saved."
    return args

def main():
    args = OTA_parser()
    ckpt = args.ckpt
    OTA_asyncor(*OTA_asyncor.util_make_dir(ckpt), **args)



    return

if __name__ == "__main__":
    pass
