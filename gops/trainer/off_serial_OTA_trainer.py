import time
import torch
import os
from gops.utils.OTA.OTA_asyncor import OTA_asyncor
from gops.trainer.off_serial_idsim_trainer import OffSerialIdsimTrainer
from torch.utils.tensorboard import SummaryWriter

from gops.utils.OTA.traj.utils import parse_csv_to_trajectory
from gops.utils.common_utils import ModuleOnDevice
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData
from gops.trainer.off_serial_trainer import OffSerialTrainer
from gops.trainer.idsim_train_evaluator import idsim_tb_tags_dict
from gops.env.env_gen_ocp.resources.lasvsim.lasvsim_env_qianxing import timeit
from cmath import inf
import tqdm

class OffSerialOTATrainer():
    def __init__(self, alg, buffer, ota,**kwargs):
        self.alg = alg
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"].startswith("prioritized") # FIXME: hard code

        # create center network
        self.networks = self.alg.networks
        self.networks.eval()

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            # self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"],map_location=torch.device('cpu')))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        # self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        # self.writer.flush()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        self.OTAasync = ota

        # self.offline_sample()

    def offline_sample(self, exps=None):
        if exps is None:
            file_path = self.OTAasync.local_get_traj()
            exps = parse_csv_to_trajectory(file_path)

        self.buffer.add_batch(exps)
        pass

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def step(self):
        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)
        self.networks.eval()

    def step_tranverse(self, times):
        pbar = tqdm.tqdm(range(times))
        for i in pbar:
            self.step()
        return


    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        # self.writer.flush()

    pass
