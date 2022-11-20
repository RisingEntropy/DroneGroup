# -*- encoding: utf-8 -*-
'''
@File    :   noise.py    
@Contact :   haoyu_deng@std.uestc.edu.cn


@Modify Time      @Author          @Version    @Desciption
------------      -------------    --------    -----------
2022/11/20 18:19   risingentropy      1.0         None
'''
import logging

import numpy.random

import LibInfo

try:
    import numpy as np
except BaseException as e:
    logging.info(f"numpy is not successfully installed, numpy is necessary for {LibInfo.LIBNAME}")
    exit(-1)

try:
    import torch
except BaseException as e:
    logging.info(f"pytorch is recommended for {LibInfo.LIBNAME}")

torch_device = LibInfo.TORCH_DEVICE


def setTorchDeivce(device: str):
    """

    :param device: if the backend is pytorch, specify the device to store the generated noise
    :return:
    """
    global torch_device
    torch_device = device
    LibInfo.TORCH_DEVICE = device


class NoiseGeneratorBase:
    def __init__(self):
        """

        """

    def getNoise(self, shape: tuple, backend='numpy', device=None):
        raise NotImplemented(
            "NoiseGeneratorBase is the super class of all noise generator, and is an abstract class that you cannot "
            "invoke getNoise method")


class GaussianNoiseGenerator(NoiseGeneratorBase):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()
        self.mean = 0
        self.sigma = 1

    def __init__(self, mean, sigma):
        """

        :param mean: mean value for the gaussian noise
        :param sigma:
        """
        super(GaussianNoiseGenerator, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def getNoise(self, shape: tuple, backend='numpy', device=None):
        """

        :param shape:
        :param backend: which backend to utilize, numpy or pytorch
        :return:
        """
        if backend == 'numpy':
            return numpy.random.normal(self.mean, self.sigma, size=shape)
        elif backend == 'pytorch':
            global torch_device
            return torch.normal(mean=self.mean, std=self.sigma, size=shape,
                                device=device if device is not None else torch_device)
        else:
            raise ValueError(f"Unknown backend:{backend}, supported backend is numpy and pytorch")


class MixedGaussionNoiseGenerator(NoiseGeneratorBase):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()
        self.mean = (0,)
        self.sigma = (1,)

    def __init__(self, mean: tuple, sigma: tuple):
        """

        :param mean: tuple, mean values for the gaussian noise
        :param sigma:
        """
        super(MixedGaussionNoiseGenerator, self).__init__()
        if len(mean) != len(sigma):
            raise ValueError(
                f"the length of parameter mean is {len(mean)},which does not match the parameter sigma({len(sigma)})")
        self.mean = mean
        self.sigma = sigma

    def setMean(self, mean: tuple):
        """

        :param mean:
        :return:
        """
        if len(mean) != len(self.mean):
            raise ValueError(
                f"the length of parameter mean is {len(mean)},which does not match the shape of former one({len(self.mean)})")
        self.mean = mean

    def setSigma(self, sigma: tuple):
        """

        :param sigma:
        :return:
        """
        if len(sigma) != len(self.sigma):
            raise ValueError(
                f"the length of parameter mean is {len(sigma)},which does not match the shape of former one({len(self.sigma)})")
        self.sigma = sigma

    def setMeanAndSigma(self, mean: tuple, sigma: tuple):
        """

        :param mean:
        :param sigma:
        :return:
        """
        if len(mean) != len(sigma):
            raise ValueError(
                f"the length of parameter mean is {len(mean)},which does not match the parameter sigma({len(sigma)})")
        self.mean = mean
        self.sigma = sigma

    def getNoise(self, shape: tuple, backend='numpy', device=None):
        """

        :param shape:
        :param backend: which backend to utilize, numpy or pytorch
        :return:
        """
        if backend == 'numpy':
            result = numpy.zeros(shape=shape)
            weight_map = numpy.random.randint(0, len(self.mean), size=shape)
            for i in range(0, len(self.mean)):
                result = result + (weight_map == i) * numpy.random.normal(self.mean[i], self.sigma[i], size=shape)
            return result

        elif backend == 'pytorch':
            result = torch.zeros(size=shape, device=device if device is not None else torch_device)
            weight_map = torch.randint(0, len(self.mean), size=shape,
                                       device=device if device is not None else torch_device)
            for i in range(0, len(self.mean)):
                result = result + (weight_map == i) * torch.normal(mean=self.mean[i], std=self.sigma[i], size=shape)
            return result
        else:
            raise ValueError(f"Unknown backend:{backend}, supported backend is numpy and pytorch")


# x = MixedGaussionNoiseGenerator((1, 2, 3), (3, 2, 1))
# print(x.getNoise((1, 2, 3, 4)))
