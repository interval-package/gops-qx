#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Monte Carlo Sampler
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-03-05, Wenxuan Wang: add action clip
#  Update Date: 2023-07-22, Zhilong Zheng: inherit from BaseSampler


from typing import List, Union
from gops.env.env_gen_ocp.resources.idsim_model.lasvsim_env_qianxing import timeit
from gops.trainer.sampler.base import BaseSampler, Experience


class OffSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        *,
        index=0, 
        noise_params=None,
        env_options: Union[dict, List[dict]] = None,
        **kwargs
    ):
        super().__init__(
            sample_batch_size,
            index = index, 
            noise_params = noise_params,
            env_options = env_options,
            **kwargs
        )
    
    
    def _sample(self) -> List[Experience]:
        batch_data = []
        for _ in range(self.horizon):
            experiences = self._step()
            batch_data.extend(experiences)
        return batch_data