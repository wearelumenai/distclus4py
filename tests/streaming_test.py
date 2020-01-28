import time
import unittest
import numpy as np

from distclus.streaming import Streaming
from tests.util import sample, rmse, nan


class TestStreaming(unittest.TestCase):

    def setUp(self):
        self.data = sample(10, 2)

    def test_streaming(self):
        algo = Streaming(sigma=.3, seed=1367098323)
        algo.push(self.data[:1])
        algo.play()
        algo.push(self.data[1:])
        time.sleep(.3)
        self.check_online(algo)
        algo.close()
        self.assertGreater(algo.max_distance, 10.)

    def test_nan(self):
        data = nan()
        self.assertRaises(ValueError, Streaming, data=data)

        algo = Streaming()
        self.assertRaises(ValueError, algo.fit, data=data)
        self.assertRaises(ValueError, algo.predict, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.push, data=data)

    def test_combine(self):
        algo = Streaming()
        combined = algo.combine(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            weight1=2
        )
        self.assertTrue(np.array_equal(combined, [2, 3, 4]))

    def test_dist(self):
        algo = Streaming()
        dist = algo.dist(
            np.array([1, 2, 3]),
            np.array([4, 5, 6])
        )
        self.assertEqual(dist, 5.196152422706632)

    def check_online(self, algo):
        centroids, labels = algo.predict_online(self.data)
        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)
