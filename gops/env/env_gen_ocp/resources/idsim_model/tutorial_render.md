# 千行环境GOPS render使用说明

## 基本概述

出于对训练结果的描述的需求，我们在本地对千行云端的运行得到的感知与结果，在本地进行可视化的渲染以及输出。

支持在测试时与训练时都进行渲染。（建议在测试时使用，训练时会导致训练速度变慢、存储空间消耗）

### 细节介绍

为了保证输出的稳定性，我们在`lasvsimEnv`中每次调用`get_obs`便会更新这个获取的内容。同时，我们会使用外部函数，获得额外的log信息。

同时在`class idSimEnv(Env):`中，我们使用如下函数进行每一步的地图渲染 

```python
self.server.env.update_render_info(self._info)
self.server.env._render_sur_byobs()
```

## 基本使用流程

基本使用流程如下：

1. 运行`example_run/run_idsim.py`
   1. 注意要选择目标要进行的策略
2. 我们进入到gops的策略调用阶段
   1. 每一帧的信息，都会被导入到`draw_qianxing`
   2. 在这个目录下，会按照测试脚本的时间，生成对应的文件夹，对应的每一个帧，都在里面，由一个图片组成
3. -运行`generate_gif.py`生成视频-
   1. 注意修改目标文件夹为存放帧信息的文件夹
   2. 目前默认时生成为mp4视频，也可以生成gif
   3. 生成的视频在原存放信息的文件夹下
4. 自动保存视频
   1. 现在已经支持自动保存视频，在运行过后，会自动调用3中的脚本，进行render视频的保存


### 选择打印的信息内容

在`lasvsimEnv`中，我们维护如下变量，我们会将如下变量的信息输出在我们的可视化界面中。

```python
    _render_tags = [
        'env_tracking_error', 
        'env_speed_error', 
        'env_delta_phi', 
        # 'category', 
        # 'env_pun2front', 
        # 'env_pun2side', 
        # 'env_pun2space', 
        # 'env_pun2rear', 
        'env_scaled_reward_part1', 
        'env_reward_collision_risk', 
        'env_scaled_pun2front', 
        'env_scaled_pun2side', 
        'env_scaled_pun2space', 
        'env_scaled_pun2rear', 
        'env_scaled_punish_boundary', 
        # 'state', 
        # 'constraint', 
        # 'env_reward_step', 
        # 'env_reward_steering', 
        # 'env_reward_acc_long', 
        # 'env_reward_delta_steer', 
        # 'env_reward_jerk', 
        # 'env_reward_dist_lat', 
        # 'env_reward_vel_long', 
        # 'env_reward_head_ang', 
        # 'env_reward_yaw_rate', 
        'env_scaled_reward_part2', 
        'env_scaled_reward_step', 
        'env_scaled_reward_dist_lat', 
        'env_scaled_reward_vel_long', 
        'env_scaled_reward_head_ang', 
        'env_scaled_reward_yaw_rate', 
        'env_scaled_reward_steering', 
        'env_scaled_reward_acc_long', 
        'env_scaled_reward_delta_steer', 
        'env_scaled_reward_jerk', 
        'total_reward',
        # 'reward_details', 
        # 'reward_comps'
        ]
    
```

如果需要修改，则注释化，或取消注释化对应的标签。

## 输出结果介绍

基本的输出结果如下所示



左侧为我们当前状态与奖励信息，右侧为我们的地图信息

红色点为轨迹信息，周车为蓝色方框

图像显示以当前自车为中心，范围由`draw_bound = 70`变量维护