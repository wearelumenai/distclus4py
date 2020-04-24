import time
import unittest
import numpy as np

from distclus import Streaming, MCMC, KMeans
from tests.util import sample, rmse, nan


def get_algo_data(algo, space, dim):
    algo = algo(space=space)

    dim = tuple(5 for _ in range(dim))

    arr1 = np.full(dim, 10)
    arr2 = np.full(dim, 20)

    return algo, arr1, arr2


class TestSpace(unittest.TestCase):

    def test(self):
        for algo in [Streaming, MCMC, KMeans]:
            for i, space in enumerate(['euclid', 'series']):
                dim = i + 1
                _algo, arr1, arr2 = get_algo_data(algo, space, dim)
                self.check_dist(_algo, arr1, arr2)
                self.check_combine(_algo, arr1, arr2)

    def check_dist(self, algo, arr1, arr2):
        dist = algo.dist(arr1, arr1)
        self.assertEqual(dist, 0)

        dist = algo.dist(arr1, arr2)
        self.assertNotEqual(dist, 0)

    def check_combine(self, algo, arr1, arr2):
        combine = algo.combine(arr1, arr1)
        self.assertTrue(np.array_equal(combine, arr1))

        combine = algo.combine(arr1, arr1, weight1=2, weight2=2)
        self.assertTrue(np.array_equal(combine, arr1))

        combine = algo.combine(arr1, arr1, weight1=2)
        self.assertTrue(np.array_equal(combine, arr1))
        self.assertFalse(np.array_equal(combine, arr2))

        combine = algo.combine(arr1, arr1, weight2=2)
        self.assertTrue(np.array_equal(combine, arr1))

        combine = algo.combine(arr1, arr2)
        self.assertFalse(np.array_equal(combine, arr1))
        self.assertFalse(np.array_equal(combine, arr2))
