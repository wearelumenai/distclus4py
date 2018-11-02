import unittest

from distclus import bind
from distclus.ffi import lib
from distclus.tests import C, tffi


class TestBind(unittest.TestCase):

    def test_initializer(self):
        self.assertEqual(lib.I_RANDOM, bind.initializer("random"))
        self.assertEqual(lib.I_GIVEN, bind.initializer("given"))
        self.assertEqual(lib.I_KMEANSPP, bind.initializer("kmeanspp"))

    def test_to_array_1d(self):
        data = self.alloc_double_array(20)

        for i in range(20):
            data[i] = float(i)

        ptr = bind.Array1D(addr=data, l1=20)
        arr = bind.to_managed_1d_array(ptr)
        self.assertEqual(20, len(arr))

        for i, value in enumerate(arr):
            self.assertEqual(float(i), value)

    def test_to_array_2d(self):
        data = self.alloc_double_array(40)

        for i in range(40):
            data[i] = float(i)

        ptr = bind.Array2D(addr=data, l1=20, l2=2)
        arr = bind.to_managed_2d_array(ptr)
        self.assertEqual(20, len(arr))

        for i, values in enumerate(arr):
            self.assertEqual(2, len(arr[i]))
            self.assertEqual(float(2 * i), values[0])
            self.assertEqual(float(2 * i + 1), values[1])

    def alloc_double_array(self, size):
        p = C.malloc(size * tffi.sizeof("double"))
        data = tffi.cast("double*", p)
        return data
