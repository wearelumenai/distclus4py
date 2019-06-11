import unittest

import numpy as np

from distclus import MCMC


class TestMCMC(unittest.TestCase):
    def setUp(self):
        self.data = np.concatenate(
            ((np.array(np.random.rand(10, 2), dtype=np.float64) + np.array([2, 4])),
             np.array(np.random.rand(10, 2), dtype=np.float64) + np.array([30, -15]))
        )

    def test_mcmc(self):
        algo = MCMC(init_k=2)
        self.assertTrue(algo.descr >= 1)

    def test_centroids_error(self):
        algo = MCMC(
            init_k=2, b=1, amp=0.1, mcmc_iter=5, seed=653126513379
        )
        try:
            algo.centroids()
        except RuntimeError as x:
            self.assertEqual(x.args[0], b'clustering not started')
            return
        self.fail()

    def test_push_run_centroids_predict(self):
        algo = MCMC(
            init_k=2, b=500, amp=0.01, mcmc_iter=5, seed=654126513379
        )
        algo.push(self.data[:5])
        algo.push(self.data[5:])

        err = algo.run(rasync=True)

        self.assertIsNone(err)

        labels = algo.predict(self.data)
        label0, label10 = self.check_labels(labels)

        algo.close()

        centroids = algo.centroids
        self.check_centroids(centroids, label0, label10)

    def test_fit_predict(self):
        algo = MCMC(init_k=2, b=500, amp=0.1)
        self.data = np.array([[0, 3], [0, 5], [0, 8], [15, 1], [15, 5], [15, 6]])
        algo.fit(self.data)

        labels = algo.predict(self.data)
        # label0, label10 = self.check_labels(labels)

        centroids = algo.centroids
        # self.check_centroids(centroids, label0, label10)

    def test_iterations(self):
        algo = MCMC(init_k=2, b=500, amp=0.1, mcmc_iter=5)
        algo.fit(self.data)

        self.assertEqual(5, algo.iterations)

    def test_acceptations(self):
        algo = MCMC(init_k=16, b=500, amp=0.1, mcmc_iter=5)
        algo.fit(self.data)

        self.assertLessEqual(1, algo.acceptations)

    def check_labels(self, labels):
        self.assertEqual(20, len(labels))
        label0, label10 = -1, -1
        for i, label in enumerate(labels):
            if i == 0:
                label0 = label
            elif i < 10:
                self.assertEqual(label0, label)
            elif i == 10:
                label10 = label
                self.assertNotEqual(label0, label10)
            else:
                self.assertEqual(label10, label)
        return label0, label10

    def check_centroids(self, centroids, label0, label10):
        self.assertEqual(2, len(centroids))

        mean0 = np.mean(self.data[:10], axis=0)
        mean10 = np.mean(self.data[10:], axis=0)

        centroid0 = centroids[label0]
        centroid10 = centroids[label10]

        dist0 = np.linalg.norm(mean0 - centroid0)
        dist10 = np.linalg.norm(mean10 - centroid10)

        self.assertGreater(1, dist0)
        self.assertGreater(1, dist10)

    def test_cosinus(self):
        algo = MCMC(space="cosinus", init_k=2, b=.5, amp=1)
        algo.fit(self.data)

        centroids = algo.centroids
        self.assertGreater(len(centroids), 1)
