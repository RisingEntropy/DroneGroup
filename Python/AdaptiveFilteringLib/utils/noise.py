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
except ModuleNotFoundError as e:
    logging.info(f"numpy is not successfully installed, numpy is necessary for {LibInfo.LIBNAME}")
    exit(-1)

try:
    import torch
except ModuleNotFoundError as e:
    logging.info(f"pytorch is recommended for {LibInfo.LIBNAME}")

torch_device = LibInfo.TORCH_DEVICE

def checkTorch():
    try:
        import torch
    except ModuleNotFoundError as e:
        logging.info(f"pytorch is recommended for {LibInfo.LIBNAME}")
        exit(-1)
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
            raise ValueError(f"Unknown backend:{backend}, supported backends are numpy and pytorch")


class LaplaceNoiseGenerator(NoiseGeneratorBase):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()
        self.loc = 0
        self.scale = 1

    def __init__(self, loc, scale):
        """

        :param mean: mean value for the gaussian noise
        :param sigma:
        """
        super(LaplaceNoiseGenerator, self).__init__()
        self.loc = loc
        self.scale = scale

    def getNoise(self, shape: tuple, backend='numpy', device=None):
        """

        :param shape:
        :param backend: which backend to utilize, numpy or pytorch
        :return:
        """
        if backend == 'numpy':
            return numpy.random.laplace(self.loc, self.scale, size=shape)
        elif backend == 'pytorch':
            global torch_device
            return torch.tensor(numpy.random.laplace(self.loc, self.scale, size=shape),
                                device=device if device is not None else torch_device)
        # here we are not using torch.from_numpy to avoid some potential problems

        else:
            raise ValueError(f"Unknown backend:{backend}, supported backends are numpy and pytorch")


class MixedNoiseGenerator(NoiseGeneratorBase):
    def __init__(self, noiseGenerators: tuple, weights: tuple):
        """

        :param noiseGenerators:
        :param weight:
        """
        super().__init__()
        if len(noiseGenerators) != len(weights):
            raise ValueError("the length of weight doesn't match noiseGenerator!")
        self.noiseGenerators = noiseGenerators
        self.weights = list(weights)
        self.wei_sum = sum(weights)

        for i in range(0, len(self.weights)):
            if self.weights[i] < 0:
                raise ValueError("weight cannot be negative, the input weights are:" + str(weights))

            self.weights[i] = self.weights[i] / self.wei_sum
        self.prefix_sum = self.weights
        for i in range(1, len(self.weights)):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + self.prefix_sum[i]

    def getNoise(self, shape: tuple, backend='numpy', device=None):
        """

        :param shape:
        :param backend:
        :param device:
        :return:
        """
        if backend == 'numpy':
            result = numpy.zeros(shape=shape)
            weight_map = numpy.random.rand(*shape)
            result = result + (weight_map < self.prefix_sum[0]) * self.noiseGenerators[0].getNoise(shape=shape,
                                                                                                backend='numpy')

            for i in range(1, len(self.noiseGenerators)):
                result = result + numpy.logical_and(weight_map >= self.prefix_sum[i - 1],weight_map < self.prefix_sum[i]) * \
                         self.noiseGenerators[i].getNoise(shape=shape, backend='numpy')
            return result

        elif backend == 'pytorch':
            result = torch.zeros(size=shape, device=device if device is not None else torch_device)

            weight_map = torch.rand(size=shape,
                                    device=device if device is not None else torch_device)  # uniform distribution

            result = result + (weight_map < self.weights[0]) * \
                     self.noiseGenerators[0].getNoise(shape=shape, backend='pytorch',
                                                      device=device if device is not None else torch_device)

            for i in range(1, len(self.noiseGenerators)):
                result = result + (torch.logical_and(weight_map >= self.prefix_sum[i - 1],weight_map < self.prefix_sum[i])) * \
                         self.noiseGenerators[i].getNoise(shape=shape, backend='pytorch',
                                                          device=device if device is not None else torch_device)
            return result
        else:
            raise ValueError(f"Unknown backend:{backend}, supported backends are numpy and pytorch")


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
y = MixedNoiseGenerator((GaussianNoiseGenerator(1,0),GaussianNoiseGenerator(10,0)),(1,1)).getNoise((100,),backend="pytorch",device="cuda:0").cpu().numpy()
import matplotlib.pyplot as plt

x = numpy.linspace(1, len(y), len(y))
plt.plot(x, y)
plt.show()
