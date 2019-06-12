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
        labels = algo.predict(self.data)
        centroids = algo.centroids
        self.assertLessEqual(rmse(self.data, centroids, labels), 2.)

    def test_online(self):
        algo = MCMC(space='series', init_k=2, b=500, amp=.1, seed=353875342)
        algo.push(self.data[:5])
        with algo.run():
            algo.push(self.data[5:])
            time.sleep(.3)
            centroids, labels = algo.predict_online(self.data)

        self.assertLessEqual(rmse(self.data, centroids, labels), 2.)
