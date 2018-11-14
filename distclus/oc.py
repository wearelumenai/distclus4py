import random
import weakref

from . import bind


class OnlineClust:
    """
    Base class for algorithm implementation using a native library
    """

    def __init__(self, lib, descr, init="kmeanspp", seed=None, *args):
        self.lib = lib
        self.seed = seed or random.randint(0, 2 ** 63)
        self.descr = descr(*args, bind.initializer(init), self.seed)

    def __finalize(self):
        weakref.finalize(self, lambda: self.lib.Free(descr))

    def fit(self, data):
        """Execute sequentially push, run and close methods.

        :param data: data to process
        """
        self.push(data)
        self.run()
        self.close()

    def push(self, data):
        """Push input data to process.

        :param data: data to push in the algorithme.
        """
        arr, l1, l2 = bind.to_c_2d_array(data)
        self.lib.Push(self.descr, arr, l1, l2)

    def run(self, rasync=False):
        """Execute (a-)synchronously the alogrithm

        :param bool rasync: Asynchronous execution if True. Default is False.
        """
        self.lib.Run(self.descr, 1 if rasync else 0)

    def close(self):
        """Close algorithm execution."""
        self.lib.Close(self.descr)

    def predict(self, data, push=False):
        """Predict """
        arr, l1, l2 = bind.to_c_2d_array(data)
        result = self.lib.Predict(self.descr, arr, l1, l2, 1 if push else 0)
        return bind.to_managed_1d_array(result)

    def centroids(self):
        """Get centroids."""
        result = self.lib.RealCentroids(self.descr)
        return bind.to_managed_2d_array(result)
