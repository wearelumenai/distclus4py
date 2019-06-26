import unittest

import numpy as np

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
            self.check_online(algo, batch_data)

    def test_frame(self):
        algo = Batch(MCMC, frame_size=10, init_k=2, b=500, amp=0.1, mcmc_iter=10)
        algo.run()
        for b in chunker(self.data, 5):
            batch_data = np.array(b)
            algo.push(batch_data)
            self.check_online(algo, batch_data)

    def test_t(self):

        data = np.concatenate((
            np.array(np.random.rand(300, 2), dtype=np.float64) + np.array([2, 4]),
            np.array(np.random.rand(300, 2), dtype=np.float64) + np.array([10, 9]),
            np.array(np.random.rand(300, 2), dtype=np.float64) + np.array([18, 20])
        ))

        for i in range(0, 400, 50):
            np.random.shuffle(data[i:i + 500])

        data = data.reshape((50, 18, 2))

        algo = Batch(MCMC, frame_size=100, b=1, amp=1, seed=136363137)
        algo.run()
        for chunk in data:
            algo.push(chunk)

        algo.predict_online(data[:1])

    def check_online(self, algo, batch_data):
        centroids, labels = algo.predict_online(batch_data)
        self.assertLessEqual(rmse(batch_data, centroids, labels), 1.)
