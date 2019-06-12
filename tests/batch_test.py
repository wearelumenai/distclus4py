import numpy as np
import unittest

from distclus import MCMC
from distclus.batch import Batch, chunker
from tests.util import sample, rmse


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.data = iter(sample(10, 2))

    def test_batch(self):
        algo = Batch(MCMC, init_k=2, b=500, amp=0.1, mcmc_iter=10)
        algo.run()
        for b in chunker(self.data, 5):
            batch_data = np.array(b)
            algo.push(batch_data)
            self.check_static(algo, batch_data)

    def check_static(self, algo, batch_data):
        labels = algo.predict(batch_data)
        centroids = algo.centroids
        self.assertLessEqual(rmse(batch_data, centroids, labels), 1.)


