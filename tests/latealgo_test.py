import time
import unittest
from threading import Thread

import numpy as np

from distclus import MCMC
from distclus.latealgo import LateAlgo


class TestLateInit(unittest.TestCase):

    def setUp(self):
        self.data = np.concatenate(
            ((np.array(np.random.rand(100, 2), dtype=np.float64) + np.array([2, 4])),
             np.array(np.random.rand(100, 2), dtype=np.float64) + np.array([30, -15]))
        )
        np.random.shuffle(self.data)

    def build(self, data):
        if len(data) >= 2:
            return MCMC(dim=len(data[0]), init_k=2, mcmc_iter=20, seed=166348259467)

    def test_build(self):
        late = LateAlgo(self.build)
        self.assertIsNone(late._algo)

    def test_push(self):
        late = LateAlgo(self.build)
        late.push(self.data[0:1])
        self.assertIsNone(late._algo)
        self.assertEqual(1, len(late._buffer))
        late.push(self.data[1:2])
        self.assertIsNotNone(late._algo)
        self.assertEqual(0, len(late._buffer))
        algo = late._algo
        late.push(self.data[2:3])
        self.assertIs(algo, late._algo)

    def test_run_sync(self):
        late = LateAlgo(self.build)
        self.assertRaises(ValueError, late.run, False)
        late.push(self.data)
        late.run(False)

    def test_run_async(self):
        late = LateAlgo(self.build)
        late.run(True)
        self.assertIsNone(late._algo)
        late.push(self.data)
        late.close()

    def test_centroids(self):
        late = LateAlgo(self.build)
        self.assertRaises(ValueError, lambda: late.centroids)
        late.run(True)
        late.push(self.data)
        time.sleep(.3)
        self.assertGreaterEqual(len(late.centroids), 2)
        late.close()

    def test_predict(self):
        late = LateAlgo(self.build)
        late.push(self.data)
        late.run(False)
        centroids = late.centroids
        labels = late.predict(self.data)
        mse = 0.
        for i, label in enumerate(labels):
            mse += np.linalg.norm(centroids[label] - self.data[i]) / len(self.data)
        self.assertLessEqual(mse, 1.)
        late.close()

    def test_predict_online(self):
        late = LateAlgo(self.build)
        late.run(True)
        self.assertRaises(ValueError, late.predict, self.data)
        late.push(self.data)
        time.sleep(.3)
        centroids, labels = late.predict_online(self.data)
        mse = 0.
        for i, label in enumerate(labels):
            mse += np.linalg.norm(centroids[label] - self.data[i]) / len(self.data)
        self.assertLessEqual(mse, 1.)
        late.close()

    def test_push_parallel(self):
        late = LateAlgo(self.build)
        late.run(True)
        threads = []
        for d in self.data:
            t = Thread(target=late.push, args=([d],))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        late.close()
