import time
import unittest
import numpy as np

from distclus.streaming import Streaming


class TestStreaming(unittest.TestCase):

    def setUp(self):
        self.data = np.concatenate((
            np.array(np.random.rand(100, 2), dtype=np.float64) + np.array([2, 4]),
            np.array(np.random.rand(100, 2), dtype=np.float64) + np.array([30, -15]),
            np.array(np.random.rand(100, 2), dtype=np.float64) + np.array([200, 150])
        ))
        np.random.shuffle(self.data)

    def test_streaming(self):
        algo = Streaming(lambd=5)
        algo.push(self.data[:1])
        algo.run(rasync=True)
        algo.push(self.data[1:])
        algo.close()
        time.sleep(1)
        centroids, labels = algo.predict_online(self.data)
        self.assertLessEqual(3, len(centroids))
        for i, label in enumerate(labels):
            d = np.abs(centroids[label] - self.data[i])
            self.assertLess(np.sum(d), 15)
