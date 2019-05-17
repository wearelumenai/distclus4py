import unittest
import numpy as np

from distclus.streaming import Streaming


class TestStreaming(unittest.TestCase):

    def setUp(self):
        self.data = np.concatenate(
            ((np.array(np.random.rand(10, 2), dtype=np.float64) + np.array([2, 4])),
             np.array(np.random.rand(10, 2), dtype=np.float64) + np.array([30, -15]))
        )
        np.random.shuffle(self.data)


    def test_streaming(self):
        algo = Streaming(lambd=5)
        algo.push(self.data[:1])
        algo.run(rasync=True)
        algo.push(self.data[1:])
        algo.close()
        centroids = algo.centroids
        self.assertLessEqual(2, len(centroids))
        labels = algo.predict(self.data)
        for i, label in enumerate(labels):
            d = np.abs(centroids[label]-self.data[i])
            self.assertLess(np.sum(d), 15)
