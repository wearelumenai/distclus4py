import unittest
import numpy as np
from distclus_bind import kmeans, mcmc


class TestsBindings(unittest.TestCase):
    def setUp(self):
        self.arr = np.concatenate(
            ((np.array(np.random.rand(10, 2), dtype=np.float64) + 2),
             np.array(np.random.rand(10, 2), dtype=np.float64) + 30))

    def test_kmeans_rand(self):
        res = kmeans(self.arr, 2, 10)
        self.assertEqual(len(res), 2)

    def test_kmeans_kmeanspp(self):
        res = kmeans(self.arr, 2, 10, initializer="kmeans++")
        self.assertEqual(len(res), 2)

    def test_kmeans_given(self):
        res = kmeans(self.arr, 2, 10, initializer="given")
        self.assertEqual(len(res), 2)

    def test_mcmc(self):
        res = mcmc(self.arr, init_k=2)
        self.assertEqual(len(res), 2)
