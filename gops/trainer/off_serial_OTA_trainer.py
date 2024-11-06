import torch
import os
from gops.utils.OTA.OTA_asyncor import OTA_asyncor
from gops.trainer.off_serial_idsim_trainer import OffSerialIdsimTrainer

class OffSerialOTATrainer(OffSerialIdsimTrainer):
    def __init__(self, alg, sampler, buffer, evaluator, OTA_config, **kwargs):
        super().__init__(alg, sampler, buffer, evaluator, **kwargs)
        model_dir = os.path.join(self.save_folder, "apprfunc")
        traj_dir = os.path.join(self.save_folder, "traj")
        onnx_dir = os.path.join(self.save_folder, "onnx")
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir)
        self.OTAserver = OTA_asyncor(model_dir=model_dir, traj_dir=traj_dir, onnx_dir=onnx_dir)

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    pass
