import time
import unittest

from distclus import KMeans
from tests.util import sample, rmse, nan


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

        algo.play()
        algo.push(self.data[5:])
        time.sleep(.3)
        self.check_online(algo)
        algo.close()

    def test_fit_predict(self):
        algo = KMeans(k=2)
        algo.fit(self.data)

        self.check_static(algo)

    def test_iterations(self):
        algo = KMeans(k=2, nb_iter=5)
        algo.fit(self.data)

        self.assertEqual(5, algo.iterations)

    def test_nan(self):
        data = nan()
        self.assertRaises(ValueError, KMeans, data=data)

        algo = KMeans()
        self.assertRaises(ValueError, algo.fit, data=data)
        self.assertRaises(ValueError, algo.predict, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.predict_online, data=data)
        self.assertRaises(ValueError, algo.push, data=data)


    def check_static(self, algo):
        labels = algo.predict(self.data)
        centroids = algo.centroids
        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)

    def check_online(self, algo):
        centroids, labels = algo.predict_online(self.data)
        self.assertLessEqual(rmse(self.data, centroids, labels), 1.)
