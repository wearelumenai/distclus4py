import unittest

import numpy as np

from distclus import MCMC


class TestSeries(unittest.TestCase):
    def setUp(self):
        np.random.seed(3458709754)
        self.data = np.concatenate(
            ((np.array(np.random.rand(10, 10, 2), dtype=np.float64) + np.array([2, 4])),
             np.array(np.random.rand(10, 10, 2), dtype=np.float64) + np.array([30, -15]))
        )

    def test_mcmc(self):
        algo = MCMC(space='series', init_k=2, b=500, seed=353875342)
        algo.fit(self.data)

        self.assertEqual(2, len(algo.centroids))
        labels = algo.predict(self.data)
        for i in range(1, 10):
            self.assertEqual(labels[0], labels[i])
        for i in range(11, 20):
            self.assertEqual(labels[10], labels[i])
        algo.close()
