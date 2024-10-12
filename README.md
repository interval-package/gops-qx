# GOPS client 使用文档

[TOC]

## `gops_client`环境详细配置

> 注意：需要从零配置环境，如果单个包手动安装的话`dependency`会容易出现错误。

对于安装`gops_client`环境，需要以下步骤：

###### install miniconda

`wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_23.10.0-1-Linux-x86_64.sh`


### `conda`装源

在主目录中查找`~/.condarc`，如没有则创建。修改内容为以下

```apl
channels:
  - conda-forge

custom_channels:
  conda-forge: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud

channel_priority: strict
show_channel_urls: true
```

### 安装`mamba`

推荐使用，速度较快。使用以下指令安装

```shell
conda install mamba
```

### 配置环境

> 注意，如果之前有安装过`gops`其他版本的话，要记得修改`gops_client.yml`中开头的地方的名称，后续使用中注意区分
>
> ```yml
> name: gops_client
> ```
> 或者也可以先删去原来安装的gops_client环境
> ```
> conda activate base
> conda env remove -n gops_client
> ```

之后，使用mamba进行环境的安装：

```shell
mamba env create -f gops_client.yml
```

> 注意这里的话一定要使用新环境建立，`mamba`会较好地处理同个`yml`中的冲突错误

然后安装当前工作目录下内容为`gops`库：

```shell
pip install -e .
```

> 注意要在`/gops_gq_xxxx_vX.Y`目录下

### 运行

**进入`gops_gq_xxxx_vX.Y`目录下**，打开终端，执行指令：

```shell
conda activate gops_client
```
多车道训练
```shell
python example_train/dsac/dsact_pi_idsim_multilane_vec_offserial.py
```
十字路口训练
```shell
python example_train/fhadp/fhadp2_attention_idsim_crossroad_offsync.py
```


# TODO:
1. 在idsim(qianxing)千行环境中做evaluation，复现国建哥在idsim(sumo)中的评估结果
2. 将SEPT接入环境