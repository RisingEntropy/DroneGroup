import os
import utils.comparator
import algorithms.Algorithm
from utils import noise
from utils import  criterions
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
n = 10
L = 1000
cmpor = utils.comparator.PlainComparator(algos=((algorithms.Algorithm.NLMS(0.1, n, L), "NLMS")
                                                , (algorithms.Algorithm.LMS(0.1, n, L), "LMS")))

cmpor.compareAndPlot(noiseGenerator=noise.GaussianNoiseGenerator(mean=0, sigma=1), n=n, L=L, criterion=criterions.getNorm(),
                     title="Plot")

