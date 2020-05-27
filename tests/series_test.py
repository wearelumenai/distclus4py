import time
import unittest

import numpy as np

from distclus import MCMC
from tests.util import rmse, sample


class TestSeries(unittest.TestCase):
    def setUp(self):
        np.random.seed(3458709754)
        self.data = sample(10, 10, 2)

    def test_mcmc(self):
        algo = MCMC(space='series', init_k=2, b=500, amp=.1, seed=353875342)
        algo.fit(self.data)
        self.check_static(algo)

    def test_online(self):
        algo = MCMC(space='series', init_k=2, b=500, amp=.1, seed=353875342)
        algo.push(self.data[:5])
        algo.play()
        algo.push(self.data[5:])
        time.sleep(.3)
        self.check_online(algo)
        algo.stop()

    def check_static(self, algo):
        _, labels = algo.predict(self.data)
        centroids = algo.centroids
        self.assertLessEqual(rmse(self.data, centroids, labels), 2.)

    def check_online(self, algo):
        centroids, labels = algo.predict(self.data)
        self.assertLessEqual(rmse(self.data, centroids, labels), 2.)
