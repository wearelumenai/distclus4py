import unittest

from distclus import bind
from distclus.ffi import lib
from tests import C, tffi


class TestBind(unittest.TestCase):

    def test_initializer(self):
        self.assertEqual((lib.I_RANDOM, 0), bind.initializer("random"))
        self.assertEqual((lib.I_GIVEN, 0), bind.initializer("given"))
        self.assertEqual((lib.I_KMEANSPP, 0), bind.initializer("kmeanspp"))

        class a:
            def __init__(self):
                self.descr = 1

        self.assertEqual((lib.I_OC, 1), bind.initializer(a()))

    def test_space(self):
        self.assertEqual(lib.S_VECTORS, bind.space("vectors"))
        self.assertEqual(lib.S_SERIES, bind.space("series"))

    def test_oc(self):
        self.assertEqual(lib.O_KMEANS, bind.oc("kmeans"))
        self.assertEqual(lib.O_MCMC, bind.oc("mcmc"))
        self.assertEqual(lib.O_KNN, bind.oc("knn"))
        self.assertEqual(lib.O_STREAMING, bind.oc("streaming"))

    def test_to_array_1d(self):
        data = alloc_double_array(20)

        for i in range(20):
            data[i] = float(i)

        ptr = bind.Array(addr=data, l1=20)
        arr = bind.to_managed_array(ptr)
        self.assertEqual(20, len(arr))

        for i, value in enumerate(arr):
            self.assertEqual(float(i), value)

    def test_to_array_2d(self):
        data = alloc_double_array(40)

        for i in range(40):
            data[i] = float(i)

        ptr = bind.Array(addr=data, l1=20, l2=2)
        arr = bind.to_managed_array(ptr)
        self.assertEqual(20, len(arr))

        for i, values in enumerate(arr):
            self.assertEqual(2, len(values))
            self.assertEqual(float(2 * i), values[0])
            self.assertEqual(float(2 * i + 1), values[1])

    def test_to_array_3d(self):
        data = alloc_double_array(40)

        for i in range(40):
            data[i] = float(i)

        ptr = bind.Array(addr=data, l1=5, l2=4, l3=2)
        arr = bind.to_managed_array(ptr)
        self.assertEqual(5, len(arr))

        for i, row in enumerate(arr):
            self.assertEqual(4, len(row))
            for j, values in enumerate(row):
                self.assertEqual(float(8 * i + 2 * j), values[0])
                self.assertEqual(float(8 * i + 2 * j + 1), values[1])


def alloc_double_array(size):
    p = C.malloc(size * tffi.sizeof("double"))
    data = tffi.cast("double*", p)
    return data
