import contextlib
import weakref

from distclus.bind import handle_error
from . import bind
from .ffi import lib, ffi

from numpy import isnan


class OnlineClust:
    """Base class for algorithm implementation using a native library"""

    def __init__(self, builder, space='vectors', data=None, *args):
        self.builder = builder
        space = bind.space(space)
        data = as_float64(data)
        data_array = bind.to_c_array(data)
        self.args = [space, *data_array, *args]
        self._set_descr()

    def _set_descr(self):
        if hasattr(self, '_OnlineClust__finalize'):
            _, free, _, _ = self.__finalize.detach()
            free()
        algo = self.builder(*self.args)
        handle_error(algo.err)
        self.descr = algo.descr
        self.__finalize = weakref.finalize(self, _make_free(self.descr))

    def fit(self, data):
        """
        Sequentially push train data, run in synchronous mode
        and close the algorithm.
        :param data: train data
        :return: centroids
        """
        self._set_descr()
        self.push(data)
        return self.batch()

    def push(self, data):
        """
        Push train data to the algorithm
        :param data: train data
        """
        data = as_float64(data)
        arr, l1, l2, l3 = bind.to_c_array(data)
        err = lib.Push(self.descr, arr, l1, l2, l3)
        handle_error(err)

    def run(self, rasync=True):
        """
        @deprecated
        Execute the algorithm in synchronous or asynchronous mode
        :param bool rasync: if True run in asynchronous mode
        otherwise run in synchronous mode (default)
        """
        return self.play() if rasync else self.batch()

    def play(self):
        """
        Play the online algorithm
        """
        err = lib.Play(self.descr)
        handle_error(err)
        return self

    def __call__(self):
        return self.play()

    def batch(self):
        """
        Batch the online algorithm
        """
        err = lib.Batch(self.descr)
        handle_error(err)
        return self.centroids

    def pause(self):
        """
        Pause the online algorithm
        """
        err = lib.Pause(self.descr)
        handle_error(err)
        return self.centroids

    def wait(self):
        """
        Wait the online algorithm
        """
        err = lib.Wait(self.descr)
        handle_error(err)
        return self.centroids

    def stop(self):
        """
        Stop the online algorithm
        """
        err = lib.Stop(self.descr)
        handle_error(err)
        return self.centroids

    def init(self):
        """
        Initializes the online algorithm
        """
        err = lib.Init(self.descr)
        handle_error(err)
        return self.centroids

    def predict(self, data):
        """
        Get labels for input data
        :param data: input data
        :return: output labels
        """
        data = as_float64(data)
        arr, l1, l2, l3 = bind.to_c_array(data)
        result = lib.Predict(self.descr, arr, l1, l2, l3)
        handle_error(result.err)
        labels = bind.Array(addr=result.labels, l1=result.n1)
        return bind.to_managed_array(labels)

    @property
    def alive(self):
        return lib.Alive(self.descr)

    @property
    def status(self):
        return ffi.string(lib.Status(self.descr))

    @property
    def centroids(self):
        """Get centroids"""
        result = lib.Centroids(self.descr)
        handle_error(result.err)
        centroids = bind.Array(
            addr=result.centroids,
            l1=result.l1, l2=result.l2, l3=result.l3
        )
        return bind.to_managed_array(centroids)

    @property
    def cluster_centers_(self):
        """Get cluster centers (equivalent to centroids attribute)"""
        return self.centroids

    def predict_online(self, data):
        """
        Get centroids and labels for input data
        :param data: input data
        :return: centroids and output labels
        """
        data = as_float64(data)
        arr, l1, l2, l3 = bind.to_c_array(data)
        result = lib.Predict(self.descr, arr, l1, l2, l3)
        handle_error(result.err)
        labels = bind.Array(
            addr=result.labels,
            l1=result.n1
        )
        centroids = bind.Array(
            addr=result.centroids,
            l1=result.l1, l2=result.l2, l3=result.l3
        )
        return bind.to_managed_array(centroids), bind.to_managed_array(labels)

    def close(self):
        """
        Close the algorithm and release resources
        """
        err = lib.Close(self.descr)
        handle_error(err)
        return self.centroids

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def __dels__(self):
        self.stop()

    def _figure(self, name):
        figure = lib.RuntimeFigure(self.descr, name)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value

    @property
    def iterations(self):
        """
        Get the number of iterations done so far
        """
        return self._figure(lib.F_ITERATIONS)

    @property
    def pushed_data(self):
        """
        Get the number of pushed data
        """
        return self._figure(lib.F_PUSHED_DATA)

    @property
    def last_iterations(self):
        """
        Get the number of last execution iterations
        """
        return self._figurer(lib.F_LAST_ITERATIONS)

    @property
    def duration(self):
        """
        Get the duration so far
        """
        return self._figure(F_DURATION)

    @property
    def last_duration(self):
        """
        Get the last execution duration
        """
        return self._figure(lib.F_LAST_DURATION)

    @property
    def last_data_time(self):
        """
        Get the last data time
        """
        return self._figure(lib.F_LAST_DATA_TIME)


def as_float64(data):
    if data is not None:
        if isnan(data).any():
            raise ValueError("data contains NaN value(s)")

        elif data.dtype != 'float64':
            data = data.astype('float64')

    return data


def _make_free(descr):
    def free():
        lib.Free(descr)

    return free
