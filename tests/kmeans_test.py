import unittest

from distclus import KMeans
from tests.util import sample, rmse


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.data = sample(10, 2)

    def test_kmeans(self):
        algo = KMeans(k=2)
        self.assertTrue(algo.descr >= 1)

    def test_push_run_centroids_predict(self):
        algo = KMeans(
            k=2, nb_iter=5, seed=653126513379
        )
        algo.push(self.data[:5])
        algo.push(self.data[5:])

        algo.run(rasync=True)

        centroids, labels = algo.predict_online(self.data)
        algo.close()

        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)

    def test_fit_predict(self):
        algo = KMeans(k=2)
        algo.fit(self.data)

        labels = algo.predict(self.data)
        centroids = algo.centroids

        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)

    def test_iterations(self):
        algo = KMeans(k=2, nb_iter=5)
        algo.fit(self.data)

        self.assertEqual(5, algo.iterations)
