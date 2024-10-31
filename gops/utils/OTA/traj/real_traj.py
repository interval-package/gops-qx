import numpy as np
import csv

from gops.env.env_gen_ocp.pyth_idsim import idSimEnv, CloudServer, IdSimModel, LasvsimEnv

class RealTraj:
    def __init__(self, env:idSimEnv):
        self.env:LasvsimEnv     = env.server.env
        self.server:CloudServer = env.server
        self.model:IdSimModel   = env.server.model
        pass

    def calc_reward(self):
        self.model.get_reward_by_state()
    pass