import time
import unittest

from distclus import MCMC
from tests.util import sample, rmse, nan

import numpy as np


class TestMCMC(unittest.TestCase):
    def setUp(self):
        self.data = sample(10, 2)

    def test_mcmc(self):
        algo = MCMC(init_k=2)
        self.assertTrue(algo.descr >= 1)

    def test_centroids_error(self):
        algo = MCMC(
            init_k=2, b=1, amp=0.1, mcmc_iter=5, seed=653126513379
        )
        try:
            algo.centroids
        except RuntimeError as x:
            self.assertEqual(x.args[0], b'clustering not started')
            return
        self.fail()

    def test_push_run_centroids_predict(self):
        algo = MCMC(
            init_k=2, b=500, amp=1, seed=654126513379
        )
        algo.push(self.data[:5])

        algo.play()
        algo.push(self.data[5:])
        time.sleep(.3)
        self.check_online(algo)
        algo.close()

    def test_context(self):
        algo = MCMC(
            init_k=2, b=500, amp=0.01, seed=654126513379
        )
        algo.push(self.data[:5])

        algo.play()
        algo.push(self.data[5:])
        time.sleep(.3)
        self.check_online(algo)
        algo.close()

    def test_fit_predict(self):
        algo = MCMC(init_k=2, b=500, amp=0.1)
        algo.fit(self.data)

        self.check_static(algo)

    def test_iterations(self):
        algo = MCMC(init_k=2, b=500, amp=0.1, mcmc_iter=5)
        algo.fit(self.data)

        self.assertEqual(5, algo.iterations)

    def test_acceptations(self):
        algo = MCMC(init_k=16, b=500, amp=0.1, mcmc_iter=5)
        algo.fit(self.data)

        self.assertLessEqual(1, algo.acceptations)

    def test_cosinus(self):
        algo = MCMC(space="cosinus", init_k=2, b=.5, amp=1)
        algo.fit(self.data)

        centroids = algo.centroids
        self.assertGreater(len(centroids), 1)

    def test_nan(self):
        data = nan()
        self.assertRaises(ValueError, MCMC, data=data)

        algo = MCMC()
        self.assertRaises(ValueError, algo.fit, data=data)
        self.assertRaises(ValueError, algo.predict, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.push, data=data)

    def test_combine(self):
        algo = MCMC()
        combined = algo.combine(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            weight1=2
        )
        self.assertTrue(np.array_equal(combined, [2, 3, 4]))

    def test_dist(self):
        algo = MCMC()
        dist = algo.dist(
            np.array([1, 2, 3]),
            np.array([4, 5, 6])
        )
        self.assertEqual(dist, 5.196152422706632)

    def check_static(self, algo):
        labels = algo.predict(self.data)
        centroids = algo.centroids
        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)

    def check_online(self, algo):
        centroids, labels = algo.predict_online(self.data)
        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)
