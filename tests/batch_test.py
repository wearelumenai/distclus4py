import numpy as np
import unittest

from distclus import MCMC
from distclus.batch import Batch, chunker
from tests.util import sample, rmse


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.data = sample(10, 2)

    def test_batch(self):
        algo = Batch(MCMC, init_k=2, b=500, amp=0.1)
        for b in chunker(self.data, 5):
            batch_data = np.array(b)
            algo.fit_batch(batch_data)
            labels = algo.predict(batch_data)
            centroids = algo.centroids
            self.assertLessEqual(rmse(batch_data, centroids, labels), 1.)


