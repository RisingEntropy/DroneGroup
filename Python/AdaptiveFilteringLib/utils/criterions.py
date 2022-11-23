# -*- encoding: utf-8 -*-
"""
@File    :   criterions.py
@Contact :   haoyu_deng@std.uestc.edu.cn


@Modify Time      @Author          @Version    @Desciption
------------      -------------    --------    -----------
2022/11/20 19:26   risingentropy      1.0         None
"""

import logging

import numpy

import LibInfo

try:
    import numpy as np
except ModuleNotFoundError as e:
    logging.info(f"numpy is not successfully installed, numpy is necessary for {LibInfo.LIBNAME}")
    exit(-1)

try:
    import torch
except ModuleNotFoundError as e:
    logging.info(f"pytorch is recommended for {LibInfo.LIBNAME}")


def Norm_torch(input1, input2):
    return torch.norm(input1 - input2)


def Norm_numpy(input1, input2):
    return numpy.linalg.norm(input1 - input2)


def getNorm(backend="numpy"):
    if backend == "numpy":
        return Norm_numpy
    elif backend == "pytorch":
        return Norm_torch
    else:
        ValueError(f"Unknown backend:{backend}, supported backend is numpy and pytorch")
