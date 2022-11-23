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

import LibInfo
import matplotlib.pyplot as plt


class ComparatorBase:
    def __init__(self, algos: tuple):
        """

        :param algos:
        """
        for algo in algos:
            if not isinstance(algo, tuple):
                raise TypeError("The algos parameter should be a tuple containing a (algorithm class:Subclass of "
                                "AlgorithmBase, name:string) tuple")
        self.algos = algos

    def compareAndPlot(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None,
                       backend="numpy", device=None, title="PlainComparatorOutput", needLog: bool = True):
        """

        :param noiseGenerator:
        :param n:
        :param L:
        :param criterion:
        :param input:
        :param weight_ori:
        :param weight_target:
        :param backend:
        :param device:
        :param title:
        :param needLog:
        :return:
        """
        raise NotImplementedError("The method in an abstract class is not invokable")

    def _compareAndPlotPytorch(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None,
                               device=None):
        """

        :param noiseGenerator:
        :param n:
        :param L:
        :param weight_ori:
        :return:
        """
        raise NotImplementedError("The method in an abstract class is not invokable")

    def _compareAndPlotNumpy(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None):
        """

        :param noiseGenerator:
        :param n:
        :param L:
        :param weight_ori:
        :return:
        """
        raise NotImplementedError("The method in an abstract class is not invokable")


class PlainComparator(ComparatorBase):

    def compareAndPlot(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None,
                       backend="numpy", device=None, title="PlainComparatorOutput", needLog: bool = True):

        if backend == "numpy":
            res = self._compareAndPlotNumpy(noiseGenerator, n, L, criterion, input, weight_ori, weight_target)
        elif backend == "pytorch":
            res = self._compareAndPlotPytorch(noiseGenerator, n, L, criterion, input, weight_ori, weight_target, device)
        else:
            raise ValueError(f"Unknown backend:{backend}, supported backends are numpy and pytorch")
        x = numpy.linspace(0, L, L).tolist()
        fig, ax = plt.subplots()
        for algo in self.algos:
            if needLog:
                ax.plot(x, numpy.log10(res[algo[1]]) * 20, label=algo[1])
            else:
                ax.plot(x, res[algo[1]], label=algo[1])
        ax.set_xlabel("iterations")
        ax.set_ylabel("error" if needLog else "error(20lgx)")
        ax.set_title(title)
        ax.legend()
        plt.show()

    def _compareAndPlotNumpy(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None,
                             ):
        result_dict = {}
        if input is None:
            input = numpy.random.normal(size=(n, L))
        if weight_ori is None:
            weight_ori = numpy.random.normal(size=(n, 1))
        if weight_target is None:
            weight_target = numpy.random.normal(size=(n, 1))

        noise = noiseGenerator.getNoise(shape=(1, L), backend="numpy")
        for algo in self.algos:
            result_dict[algo[1]] = algo[0].iterateExample(input, noise, weight_target, criterion, weight_ori=weight_ori,
                                                          backend="numpy")
        return result_dict

    def _compareAndPlotPytorch(self, noiseGenerator, n, L, criterion, input=None, weight_ori=None, weight_target=None,
                               device=None):
        result_dict = {}
        if input is None:
            input = torch.tensor(numpy.random.normal(size=(n, L)),
                                 device=device if device is not None else LibInfo.TORCH_DEVICE)
        if weight_ori is None:
            weight_ori = torch.tensor(numpy.random.normal(size=(n, 1)),
                                      device=device if device is not None else LibInfo.TORCH_DEVICE)
        if weight_target is None:
            weight_target = torch.tensor(numpy.random.normal(size=(n, 1)),
                                         device=device if device is not None else LibInfo.TORCH_DEVICE)

        noise = noiseGenerator.getNoise(shape=(1, L), backend="pytorch", device=device)
        for algo in self.algos:
            result_dict[algo[1]] = algo[0].iterateExample(input, noise, weight_target, criterion, weight_ori=weight_ori,
                                                          backend="pytorch",
                                                          device=device)
        return result_dict
