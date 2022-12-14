# -*- encoding: utf-8 -*-
"""
@File    :   Algorithm.py
@Contact :   haoyu_deng@std.uestc.edu.cn


@Modify Time      @Author          @Version    @Desciption
------------      -------------    --------    -----------
2022/11/20 18:57   risingentropy      1.0         None
"""
import logging

import LibInfo

try:
    import numpy
except ModuleNotFoundError as e:
    logging.info(f"numpy is not successfully installed, numpy is necessary for {LibInfo.LIBNAME}")
    exit(-1)

try:
    import torch
except ModuleNotFoundError as e:
    logging.info(f"pytorch is recommended for {LibInfo.LIBNAME}")


class AlgorithmBase:
    def __init__(self, n: int, data_len: int):
        """
        :param n:
        :param data_len:
        """
        self.n = n
        self.data_len = data_len

    def iterateExample(self, input, noise, ground_truth, criterion, weight_ori=None, backend="numpy"):
        """

        :param input:
        :param weight_ori:
        :param noise:
        :param ground_truth:
        :param criterion:
        :param backend:
        :return:
        """
        if backend == "numpy":
            if weight_ori is None:
                return self._iterateNumpy(input, numpy.random.normal(size=(self.n, 1)), noise, ground_truth, criterion)
            else:
                return self._iterateNumpy(input, weight_ori, noise, ground_truth, criterion)
        elif backend == "pytorch":
            if weight_ori is None:
                return self._iteratePytorch(input,
                                            torch.tensor(numpy.random.normal(size=(self.n, 1)), device=input.device),
                                            noise, ground_truth, criterion)
            return self._iteratePytorch(input, weight_ori, noise, ground_truth, criterion)
        else:
            raise ValueError(f"Unknown backend:{backend}, supported backends are numpy and pytorch")

    def _iterateNumpy(self, input, weight_ori, noise, ground_truth, criterion):
        """

        :param input:
        :param weight_ori:
        :param noise:
        :param ground_truth:
        :param criterion:
        :return:
        """
        raise NotImplementedError("AlgorithmExampleBase is an abstract class and its methods cannot be invoked")

    def _iteratePytorch(self, input, weight_ori, noise, ground_truth, criterion):
        """

        :param input:
        :param weight_ori:
        :param noise:
        :param ground_truth:
        :param criterion:
        :return:
        """
        raise NotImplementedError("AlgorithmExampleBase is an abstract class and its methods cannot be invoked")


class LMS(AlgorithmBase):
    """

    """

    def __init__(self, step_size: float, n: int, data_len: int):
        """

        :param step_size:
        :param n:
        :param data_len:
        """
        super().__init__(n, data_len)
        self.step_size = step_size

    def _iterateNumpy(self, input, weight_ori, noise, ground_truth, criterion):
        weight = weight_ori
        d_k = numpy.matmul(ground_truth.T, input) + noise
        error = []
        for k in range(0, self.data_len):
            err = d_k[:, k] - numpy.matmul(weight.T, input[:, k])
            weight = weight + self.step_size * err * input[:, k].reshape(weight.shape)
            error.append(criterion(weight, ground_truth))
        return error

    def _iteratePytorch(self, input, weight_ori, noise, ground_truth, criterion):
        weight = weight_ori
        d_k = torch.mm(ground_truth.T, input) + noise
        error = []
        for k in range(0, self.data_len):
            err = d_k[:, k] - torch.mm(weight.T, input[:, k].unsqueeze(dim=1))
            weight = weight + self.step_size * err * input[:, k].reshape(weight.shape)
            error.append(criterion(weight, ground_truth).cpu().item())
        return error


class NLMS(AlgorithmBase):
    def __init__(self, step_size: float, n: int, data_len: int):
        super().__init__(n, data_len)
        self.step_size = step_size

    def _iterateNumpy(self, input, weight_ori, noise, ground_truth, criterion):
        weight = weight_ori
        d_k = numpy.matmul(ground_truth.T, input) + noise
        error = []
        for k in range(0, self.data_len):
            err = d_k[:, k] - numpy.matmul(weight.T, input[:, k])
            weight = weight + (
                    self.step_size * err * input[:, k] / (numpy.matmul(input[:, k].T, input[:, k]))).reshape(
                weight.shape)
            error.append(criterion(weight, ground_truth))
        return error

    def _iteratePytorch(self, input, weight_ori, noise, ground_truth, criterion):
        weight = weight_ori
        d_k = torch.mm(ground_truth.T, input) + noise
        error = []
        for k in range(0, self.data_len):
            err = d_k[:, k] - torch.mm(weight.T, input[:, k].unsqueeze(dim=1))
            weight = weight + (self.step_size * err * input[:, k] / (
                torch.mm(torch.unsqueeze(input[:, k], dim=1).T, torch.unsqueeze(input[:, k], dim=1)))).reshape(
                weight.shape)
            error.append(criterion(weight, ground_truth).cpu().item())
        return error

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # cuda
# device = "cuda:0"
# n = 100
# L = 10000
# weight_ori = torch.from_numpy(numpy.random.normal(size=(n, 1))).to(device)
# weight_target = torch.from_numpy(numpy.random.normal(size=(n, 1))).to(device)
# noise = noise.LaplaceNoiseGenerator(0, 1).getNoise(shape=(1, L), backend="pytorch", device=device)
# # noise = numpy.random.normal(size=(1, L))
# input = torch.from_numpy(numpy.random.normal(size=(n, L))).to(device)
# algo = NLMS(0.1, n, L)
# res = algo.iterateExample(input, noise, weight_target, criterion=cri.getNorm("pytorch"), backend="pytorch")
# x = numpy.linspace(0, len(res), len(res))
# plt.plot(x, res)
# plt.title("n=1000")
# plt.show()

# device = "cuda:0"
# n = 100
# L = 10000
# weight_ori = numpy.random.normal(size=(n, 1))
# weight_target = numpy.random.normal(size=(n, 1))
# noise = noise.MixedGaussionNoiseGenerator((1,2,3,4,5),(5,4,3,2,1),(1,2,3,4,5)).getNoise(shape=(1,L))
# # noise = numpy.random.normal(size=(1, L))
# input = numpy.random.normal(size=(n, L))
# algo = NLMS(0.1, n, L)
# res = algo.iterate(input, weight_ori, noise, weight_target, criterion=cri.getNorm())
# x = numpy.linspace(0, len(res), len(res))
# plt.plot(x, res)
# plt.show()
